"""离线内容画像（M6）：文本 LLM 为全部电影生成结构化内容画像。

VLM 降级说明：当前 API key 无视觉模型权限（qwen-vl-* 均 403），改为
qwen3.7-plus 基于片名+类型的文本画像（实测含冷门 B 级片在内质量良好）；
VLM 视觉字段（海报视觉风格）待权限开通后由 enrich_vlm.py 补充。

输出 data/artifacts/profiles.parquet（movie_id, genres, mood, era, desc），
只存成功项，断点续跑自动重试缺失。节流 + 并发限制（沿用海报增强模式）。
用法：python scripts/enrich_profiles.py [并发=4]
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src import config
from src.llm.service import KNOWN_GENRES, _extract_json

MOODS = ("轻松,幽默,温馨,治愈,浪漫,热血,史诗,紧张,悬疑,惊悚,黑暗,沉重,"
         "科幻感,怀旧,现实,荒诞")
MIN_INTERVAL = 0.45          # 全局节流（秒）
_last_req = 0.0
_lock = threading.Lock()
_client = None


def _throttle():
    global _last_req
    with _lock:
        wait = _last_req + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req = time.time()


def _client_():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=config.LLM_API_KEY,
                         base_url=config.LLM_BASE_URL, timeout=40)
    return _client


def fetch_one(mid: int, title: str, genres: str):
    g = genres.replace("|", "/") if isinstance(genres, str) else ""
    prompt = (
        f"根据电影《{title}》（类型：{g}）输出内容画像 JSON：\n"
        '{"genres": [从这18个类型选0~3个: ' + ", ".join(KNOWN_GENRES) + '],\n'
        f'  "mood": 从[{MOODS}]中选1个,\n'
        '  "era": 年代感如"90年代"或"",\n'
        '  "desc": 一句话中文描述电影气质与题材(不超过40字)}\n'
        '如果你不认识这部电影，输出 {"unknown": true}，不要编造。只输出 JSON。')
    for attempt in range(3):
        try:
            _throttle()
            r = _client_().chat.completions.create(
                model=config.LLM_MODEL, temperature=0.2,
                extra_body={"enable_thinking": False},
                messages=[{"role": "user", "content": prompt}])
            obj = _extract_json(r.choices[0].message.content or "")
            if obj is None or obj.get("unknown"):
                return None
            gs = [g for g in (obj.get("genres") or []) if g in KNOWN_GENRES]
            if not gs and not obj.get("desc"):
                return None
            return {"movie_id": mid, "genres": "|".join(gs),
                    "mood": str(obj.get("mood", ""))[:6],
                    "era": str(obj.get("era", ""))[:8],
                    "desc": str(obj.get("desc", ""))[:60]}
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return None


def _save(out_path, results, done):
    if results:
        new = pd.DataFrame(results)
        if done:
            old = pd.read_parquet(out_path)
            new = pd.concat([old, new]).drop_duplicates("movie_id")
        new.to_parquet(out_path)


def main() -> None:
    if not config.LLM_API_KEY:
        print("未配置 LLM_API_KEY（.env）")
        return
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
    out_path = config.ART_DIR / "profiles.parquet"
    done = set()
    if out_path.exists():
        done = set(pd.read_parquet(out_path).movie_id.astype(int))
        print(f"已有 {len(done)} 部，跳过")
    todo = [(int(r.movie_id), str(r.title), str(r.genres))
            for r in movies.itertuples() if int(r.movie_id) not in done]
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"待生成 {len(todo)} 部，并发 {workers}（预计 "
          f"{len(todo) * MIN_INTERVAL / 60:.0f}~{len(todo) * 2 / workers / 60:.0f} 分钟）")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_one, m, t, g): m for m, t, g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r:
                results.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(todo)}  命中 {len(results)}  "
                      f"({time.time() - t0:.0f}s)")
                _save(out_path, results, done)
    _save(out_path, results, done)
    total = len(done) + len(results)
    print(f"完成：{total}/{len(movies)} 部有画像（{total / len(movies):.1%}）"
          f"→ {out_path}")


if __name__ == "__main__":
    main()
