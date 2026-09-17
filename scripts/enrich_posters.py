"""离线海报增强：IMDb suggestion API（大陆可达，无需 key）。

背景：TMDB API (api.themoviedb.org) 在本网络被阻断（DNS 污染 + SNI 阻断），
image.tmdb.org 图片 CDN 可达但无法获得 poster_path。改用 IMDb 建议 API
(v3.sg.media-imdb.com/suggestion/x/{query}.json)：
- 按 标题（去年份 + 冠词归位）+ 年份（±1）匹配电影；
- 海报 URL 来自 m.media-amazon.com（可达），._V1_.jpg 改写为 ._V1_UX300_.jpg
  控制尺寸；
- 输出 data/artifacts/posters.parquet（movie_id, imdb_id, poster_url），
  只存成功项，断点续跑自动重试缺失项。

用法：python scripts/enrich_posters.py [并发数=8]
"""
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src import config

YEAR_RE = re.compile(r"\s*\((\d{4})\)\s*$")
ARTICLE_RE = re.compile(
    r"^(.+), (The|A|An|Les|La|Le|Los|El|Las|Das|Der|Die|Il|Lo|Une|Un|O|Yo)$")
API = "https://v3.sg.media-imdb.com/suggestion/x/{}.json"
MIN_INTERVAL = 1.2          # 全局请求节流（秒）——超过会触发 429
_last_req = 0.0
_throttle_lock = threading.Lock()


def _throttle():
    """全局节流：保证请求起始间隔不小于 MIN_INTERVAL。"""
    global _last_req
    with _throttle_lock:
        wait = _last_req + MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req = time.time()


def split_title_year(title: str):
    m = YEAR_RE.search(title)
    year = int(m.group(1)) if m else 0
    t = YEAR_RE.sub("", title).strip()
    m2 = ARTICLE_RE.match(t)                     # "Bug's Life, A" → "A Bug's Life"
    if m2:
        t = f"{m2.group(2)} {m2.group(1)}"
    return t, year


def fetch_one(mid: int, title: str):
    t, year = split_title_year(title)
    url = API.format(urllib.parse.quote(t))
    backoff = 0
    for attempt in range(6):
        try:
            _throttle()
            with urllib.request.urlopen(url, timeout=20) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:                    # 限流：长退避
                backoff = max(backoff * 2, 30)
                time.sleep(backoff)
                continue
            return None
        except Exception:
            if attempt >= 3:
                return None
            time.sleep(1.0 * (attempt + 1))
    else:
        return None
    best = None
    for e in d.get("d", []):
        if e.get("qid") not in ("movie", None):
            continue
        if year and e.get("y") and abs(int(e["y"]) - year) <= 1:
            best = e
            break
        if best is None:
            best = e                           # 无年份匹配时兜底
    if not best or not best.get("i", {}).get("imageUrl"):
        return None
    img = re.sub(r"\._V1[^.]*\.jpg$", "._V1_UX300_.jpg",
                 best["i"]["imageUrl"])
    if img == best["i"]["imageUrl"]:            # 无 _V1 后缀则原样使用
        img = best["i"]["imageUrl"]
    return {"movie_id": mid, "imdb_id": best.get("id", ""),
            "poster_url": img}


def main() -> None:
    movies = pd.read_parquet(config.ART_DIR / "movies.parquet")
    done = set()
    out_path = config.ART_DIR / "posters.parquet"
    if out_path.exists():
        done = set(pd.read_parquet(out_path).movie_id.astype(int))
        print(f"已有 {len(done)} 部，跳过")
    # 按热度降序抓取：优先覆盖最常被推荐的电影，部分覆盖时演示可见性最好
    prio = {}
    if (config.ART_DIR / "hot.parquet").exists():
        hot = pd.read_parquet(config.ART_DIR / "hot.parquet")
        prio = dict(zip(hot.movie_id.astype(int),
                        hot.pos_count.astype(int)))
    todo = [(int(r.movie_id), str(r.title))
            for r in movies.itertuples() if int(r.movie_id) not in done]
    todo.sort(key=lambda x: -prio.get(x[0], 0))
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"待抓取 {len(todo)} 部（按热度降序），并发 {workers}")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch_one, mid, title): mid for mid, title in todo}
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
    print(f"完成：{total}/{len(movies)} 部有海报 "
          f"({total / len(movies):.1%})，输出 {out_path}")


def _save(out_path, results, done):
    if results:
        new = pd.DataFrame(results)
        if done:
            old = pd.read_parquet(out_path)
            new = pd.concat([old, new]).drop_duplicates("movie_id")
        new.to_parquet(out_path)


if __name__ == "__main__":
    main()
