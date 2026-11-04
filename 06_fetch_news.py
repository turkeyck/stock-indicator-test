"""從鉅亨網新聞列表（逐日查詢，避開 30 頁上限）抓國發會每月景氣燈號「當時公布」的新聞"""
import json, time, datetime, urllib.request, sys
from pathlib import Path
OUT = Path(__file__).parent / "raw" / "news"; OUT.mkdir(parents=True, exist_ok=True)

def get(cat, s, e, page):
    u = f"https://api.cnyes.com/media/api/v1/newslist/category/{cat}?startAt={s}&endAt={e}&limit=100&page={page}"
    for k in range(6):
        try: return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60))
        except Exception: time.sleep(2 ** k)
    print("FAILED", u, file=sys.stderr, flush=True); return None

KEY = ("景氣對策", "景氣燈號", "對策信號", "領先指標", "綜合判斷分數", "景氣概況")
y = int(sys.argv[1]); cats = ["tw_macro", "headline", "tw_stock"]
for m in range(1, 13):
    f = OUT / f"d{y}{m:02d}.json"
    if f.exists(): continue
    hits = {}; fail = 0
    day = datetime.datetime(y, m, 20); end = datetime.datetime(y + (m == 12), m % 12 + 1, 4)
    while day < end:
        s, e = int(day.timestamp()), int((day + datetime.timedelta(days=1)).timestamp()) - 1
        for cat in cats:
            p = 1
            while True:
                d = get(cat, s, e, p)
                if not d: fail += 1; break
                for x in d["items"]["data"]:
                    t = x["title"] + " " + (x.get("summary") or "")
                    if any(k in t for k in KEY) and "台經院" not in t and "製造業景氣" not in t:
                        hits[x["newsId"]] = {"id": x["newsId"], "t": x["publishAt"], "title": x["title"], "summary": x.get("summary")}
                if p >= min(d["items"]["last_page"], 30): break
                p += 1
        day += datetime.timedelta(days=1)
    f.write_text(json.dumps(list(hits.values()), ensure_ascii=False), encoding="utf-8")
    print(y, m, len(hits), "fail", fail, flush=True)
