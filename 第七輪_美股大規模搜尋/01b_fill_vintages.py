"""補抓 ALFRED 缺漏的月底版本（每次請求最多 12 個版本）"""
import io, time, urllib.request, urllib.error
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
A = Path(__file__).parent / "raw" / "alfred"
def get(u):
    for k in range(6):
        try: return urllib.request.urlopen(u, timeout=120).read().decode()
        except urllib.error.HTTPError as e:
            if e.code in (400, 404): return None
            time.sleep(2 ** k)
        except Exception: time.sleep(2 ** k)
dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("1965-01-31", "2026-10-31", freq="ME")]
for s in ["PAYEMS", "UNRATE", "INDPRO", "HOUST", "AWHMAN", "M2SL"]:
    V = pd.read_csv(A / f"{s}.csv", index_col=0)
    have = {c.split("_")[-1] for c in V.columns}
    miss = [d for d in dates if d.replace("-", "") not in have]
    if s == "M2SL": miss = [d for d in miss if d >= "1980-01-01"]
    chunks = [miss[i:i + 12] for i in range(0, len(miss), 12)]
    def job(ch):
        t = get(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={','.join([s]*len(ch))}&vintage_date={','.join(ch)}")
        if t and t.startswith("observation_date"): return [pd.read_csv(io.StringIO(t), index_col=0, na_values=["."])]
        out = []
        for d in ch:
            t = get(f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id={s}&vintage_date={d}")
            if t and t.startswith("observation_date"): out.append(pd.read_csv(io.StringIO(t), index_col=0, na_values=["."]))
        return out
    with ThreadPoolExecutor(6) as ex: got = [df for L in ex.map(job, chunks) for df in L]
    W = pd.concat([V] + got, axis=1); W = W.loc[:, ~W.columns.duplicated()]
    W = W[sorted(W.columns)]; W.to_csv(A / f"{s}.csv"); print(s, "版本數", W.shape[1], flush=True)
