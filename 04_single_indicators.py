"""各指標單獨檢定（描述性；21 項 → Bonferroni 門檻 0.05/21≈0.0024）"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
ns = {"__file__": str(HERE / "02_evaluate.py")}
exec((HERE / "02_evaluate.py").read_text(encoding="utf-8").split("lines = []")[0], ns)
sc, setup, nw_t, circ_p, avgpx = ns["sc"], ns["setup"], ns["nw_t"], ns["circ_p"], ns["avgpx"]
df = setup(avgpx, 1).loc["1992-01":]
rows = []
for c in sc.columns[3:]:
    x = sc[c].shift(1); z = pd.DataFrame({"x": x, "y": df.fwd12}).dropna()
    if len(z) < 120: continue
    _, t = nw_t(z.y.values, z.x.values); r, p = circ_p(z.y.values, z.x.values, 2000)
    zc = pd.DataFrame({"x": (x - x.shift(6)), "y": df.fwd12}).dropna(); _, tc = nw_t(zc.y.values, zc.x.values)
    rows.append((c, len(z), round(r, 3), round(t, 2), round(p, 4), round(tc, 2)))
out = pd.DataFrame(rows, columns=["指標", "N", "分數→未來12月corr", "NW t", "單尾p(正向)", "分數6月變化 NW t"]).sort_values("NW t")
print(out.to_string(index=False)); out.to_csv(HERE / "out" / "single_indicators.csv", index=False, encoding="utf-8-sig")
