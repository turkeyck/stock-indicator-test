"""追加測試（使用者建議，看過第十輪結果後才加入）：三個訊號各管 1/3 資金"""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
src = (HERE / "01_combine.py").read_text(encoding="utf-8").split("L = []")[0]
ns = {"__file__": str(HERE / "01_combine.py")}; exec(src, ns)
M, T, D, r50, r2, dates, summ, CASH, COST = (ns[k] for k in ["M", "T", "D", "r50", "r2", "dates", "summ", "CASH", "COST"])
def run_w(w, r):
    hold = w.shift(2).fillna(0.0); tr = hold.diff().abs().fillna(0)
    return hold * r + (1 - hold) * CASH - tr * COST, hold
def circ_p(hold, r, ret):
    hv, rv = hold.values, r.values; tot = np.prod(1 + ret.values); sims = []
    for k in range(60, len(hv) - 60, 3):
        hh = np.roll(hv, k); trn = np.abs(np.diff(np.r_[hh[0], hh]))
        sims.append(np.prod(1 + hh * rv + (1 - hh) * CASH - trn * COST))
    return (np.sum(np.array(sims) >= tot) + 1) / (len(sims) + 1)
VARS = {"三等分 (M+T+D)/3": (M.astype(float) + T + D) / 3,
        "兩等分 (M+T)/2": (M.astype(float) + T) / 2,
        "固定 2/3 持股（對照）": pd.Series(2 / 3, index=dates)}
L = []
def P(*a): s = " ".join(map(str, a)); print(s); L.append(s)
eps = {"2008": ("2007-10-30", "2008-11-20"), "2011": ("2011-02-08", "2011-12-19"), "2015": ("2015-04-27", "2015-08-24"),
       "2018": ("2018-01-23", "2018-10-26"), "2020": ("2020-01-14", "2020-03-19"), "2022": ("2022-01-05", "2022-10-25"), "2025": ("2025-01-21", "2025-04-09")}
for nm, r in (("0050", r50), ("正2", r2)):
    bh = summ(r); P(f"\n##### {nm}：一直持有 年化 {bh[0]:.2%}、最大回撤 {bh[1]:.1%}")
    rows = []
    for k, w in VARS.items():
        ret, hold = run_w(w, r); cg, dd = summ(ret); p = circ_p(hold, r, ret) if "對照" not in k else float("nan")
        rows.append({"規則": k, "年化": f"{cg:.2%}", "最大回撤": f"{dd:.1%}", "平均持股": f"{hold.mean():.0%}", "年換手": f"{hold.diff().abs().sum()/len(hold)*252:.1f}倍",
                     "隨機對照p": "" if np.isnan(p) else f"{p:.3f}", **{e: f"{(1+ret.loc[a:b]).prod()-1:+.0%}" for e, (a, b) in eps.items()}})
    P(pd.DataFrame(rows).to_string(index=False))
(HERE / "out" / "equal_thirds.txt").write_text("\n".join(L), encoding="utf-8")
