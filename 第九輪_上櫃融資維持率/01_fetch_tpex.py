"""逐日抓櫃買中心：個股融資餘額(張)、融資金(仟元)、個股收盤價 → raw/daily/YYYYMMDD.json.gz"""
import json, gzip, time, urllib.request, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
HERE = Path(__file__).parent; D = HERE / "raw" / "daily"
tx = json.load(open(HERE.parent / "raw" / "price" / "taiex.json"))["data"]
days = sorted({x["date"] for x in tx if x["date"] >= "2007-04-01"})
extra = pd.bdate_range("2026-11-04", "2026-11-06").strftime("%Y-%m-%d").tolist()
days = sorted(set(days) | set(extra))
def num(s):
    try: return float(str(s).replace(",", ""))
    except: return None
def get(u):
    for k in range(6):
        try: return json.loads(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read())
        except Exception: time.sleep(2 ** k)
    return None
def job(day):
    f = D / (day.replace("-", "") + ".json.gz")
    if f.exists(): return day, "skip"
    q = day.replace("-", "%2F")
    m = get(f"https://www.tpex.org.tw/www/zh-tw/margin/balance?date={q}&response=json")
    p = get(f"https://www.tpex.org.tw/www/zh-tw/afterTrading/otc?date={q}&type=EW&response=json")
    if not m or not p: return day, "FAIL"
    mt = m["tables"][0]; pt = p["tables"][0]
    if not mt.get("data"): return day, "empty"
    loan = None
    for row in mt.get("summary", []):
        if "融資金" in row[1]: loan = num(row[6])
    out = {"loan_k": loan,
           "margin": {r[0]: num(r[6]) for r in mt["data"]},
           "close": {r[0]: num(r[2]) for r in pt.get("data", [])}}
    with gzip.open(f, "wt", encoding="utf-8") as g: json.dump(out, g)
    return day, f"ok {len(out['margin'])} {len(out['close'])}"
with ThreadPoolExecutor(8) as ex:
    for i, (d, s) in enumerate(ex.map(job, days)):
        if s != "skip" and (i % 100 == 0 or s.startswith(("FAIL", "empty"))): print(i, d, s, flush=True)
print("done")
