"""下載 FRED 現行資料 + ALFRED 每月底當時版本"""
import urllib.request, time, io, sys
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
RAW = Path(__file__).parent / "raw"; (RAW / "fred").mkdir(parents=True, exist_ok=True); (RAW / "alfred").mkdir(exist_ok=True)
import urllib.error
def get(u):
    for k in range(6):
        try: return urllib.request.urlopen(u, timeout=120).read().decode()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404): return None      # 該日期沒有版本，不重試
            time.sleep(2 ** k)
        except Exception: time.sleep(2 ** k)
    return None
CUR = ["TB3MS", "GS10", "GS2", "FEDFUNDS", "AAA", "BAA", "CPIAUCNS", "WTISPLC", "PPIACO", "ICSA", "PERMIT", "UMCSENT",
       "PAYEMS", "UNRATE", "INDPRO", "HOUST", "AWHMAN", "M2SL", "T10Y3M", "BAA10Y", "VIXCLS", "DTWEXBGS", "TWEXBMTH", "NFCI"]
def cur(s):
    t = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}")
    if t: (RAW / "fred" / f"{s}.csv").write_text(t)
    return s, bool(t)
if not (RAW / "fred" / "NFCI.csv").exists():
    with ThreadPoolExecutor(6) as ex: print(list(ex.map(cur, CUR)))
RT = ["PAYEMS", "UNRATE", "INDPRO", "HOUST", "AWHMAN", "M2SL"]
dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1965-01-31", "2026-10-31", freq="ME")]
def vint(args):
    s, chunk = args
    for k in range(3):
        t = get(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={','.join([s]*len(chunk))}&vintage_date={','.join(chunk)}")
        if t and t.startswith("observation_date"): return s, chunk, t
        # 若整批失敗（早於最早版本），逐一嘗試
        if k == 0 and len(chunk) > 1: break
    out = []
    for d in chunk:
        t = get(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={s}&vintage_date={d}")
        if t and t.startswith("observation_date"): out.append(t)
    return s, chunk, out
RT = [s for s in RT if not (RAW / "alfred" / f"{s}.csv").exists()]
jobs = [(s, dates[i:i + 24]) for s in RT for i in range(0, len(dates), 24)]
res = {}
with ThreadPoolExecutor(6) as ex:
    for s, chunk, t in ex.map(vint, jobs):
        ts = [t] if isinstance(t, str) else t
        for tt in ts:
            df = pd.read_csv(io.StringIO(tt), index_col=0, na_values=["."])
            res.setdefault(s, []).append(df)
        print(s, chunk[0], len(ts), flush=True)
for s, L in res.items():
    pd.concat(L, axis=1).to_csv(RAW / "alfred" / f"{s}.csv")
    print(s, "版本數", sum(d.shape[1] for d in L))
