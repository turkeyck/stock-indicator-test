"""抓候選新聞的全文（快取在 raw/news/body/{id}.txt）"""
import json, re, glob, time, sys, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).parent; B = HERE / "raw" / "news" / "body"; B.mkdir(exist_ok=True)
KEY = ("景氣對策", "景氣燈號", "對策信號", "領先指標", "綜合判斷分數", "景氣概況")

def body(i):
    f = B / f"{i}.txt"
    if f.exists(): return
    for k in range(5):
        try:
            s = urllib.request.urlopen(urllib.request.Request(f"https://news.cnyes.com/news/id/{i}", headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read().decode("utf-8", "ignore")
            t = re.sub(r"<script.*?</script>|<style.*?</style>", "", s, flags=re.S); t = re.sub(r"<[^>]+>", "\n", t)
            t = re.sub(r"\n\s*\n+", "\n", t); a = t.find("AI新聞摘要"); a = a if a >= 0 else 0
            e = t.find("掌握全球財經資訊", a); f.write_text(t[a:e if e > 0 else a + 6000], encoding="utf-8"); return
        except Exception: time.sleep(2 ** k)
    print("FAILED", i, file=sys.stderr)

ids = []
for fn in glob.glob(str(HERE / "raw" / "news" / "d*.json")):
    for x in json.load(open(fn, encoding="utf-8")):
        t = x["title"] + (x.get("summary") or "")
        if any(k in t for k in KEY) and "預告" not in t: ids.append(x["id"])
ids = sorted(set(ids)); print("候選新聞", len(ids))
with ThreadPoolExecutor(8) as ex: list(ex.map(body, ids))
print("done", len(list(B.glob("*.txt"))))
