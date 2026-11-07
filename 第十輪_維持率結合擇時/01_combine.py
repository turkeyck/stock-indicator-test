"""維持率 × 趨勢 × 擴散 組合回測（依 00_預先登記.md）"""
import json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parent; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
mr = pd.read_csv(ROOT / "第九輪_上櫃融資維持率" / "out" / "otc_margin_ratio.csv", index_col=0, parse_dates=True).ratio
tx = pd.DataFrame(json.load(open(ROOT / "raw" / "price" / "taiex.json"))["data"])
taiex = pd.Series(tx.close.values, index=pd.to_datetime(tx.date)).astype(float)
dates = mr.index.intersection(taiex.index); mr = mr.reindex(dates)
px = taiex.reindex(dates)
txr = taiex.pct_change() + 0.03 / 252
def onidx(r):
    w = (1 + r.fillna(0)).cumprod().reindex(r.index.union(dates)).ffill().reindex(dates); return w.pct_change().fillna(0)
r50 = onidx(txr - 0.004 / 252); r2 = onidx(2 * txr - 0.02 / 252)
CASH, COST = 0.01 / 252, 0.001

# --- 訊號（皆為 t 日收盤可知）---
M = np.zeros(len(dates), bool); st = False
for i, v in enumerate(mr.values):
    if not st and v < 150: st = True
    elif st and v > 210: st = False
    M[i] = st
M = pd.Series(M, index=dates)
ma = taiex.rolling(200).mean().reindex(dates); T = (px > ma)
ld = pd.read_csv(ROOT / "raw" / "ndc" / "領先指標構成項目.csv", na_values=["-"])
eo = pd.Series(pd.to_numeric(ld.iloc[:, 1], errors="coerce").values, index=pd.PeriodIndex(ld.Date.astype(str), freq="M")).dropna()
a3 = eo.rolling(3).mean(); dsig = (a3 > a3.shift(3)).dropna()
avail = pd.Series(dsig.values, index=[(p + 1).to_timestamp() + pd.Timedelta(days=24) for p in dsig.index])
D = avail.reindex(avail.index.union(dates)).ffill().reindex(dates).fillna(False).astype(bool)

RULES = {"M 維持率": M, "T 趨勢": T, "D 擴散": D, "M且T": M & T, "M或T": M | T, "T且D": T & D, "T或D": T | D,
         "M且D": M & D, "三取二": (M.astype(int) + T.astype(int) + D.astype(int)) >= 2}

def run(sig, r):
    hold = sig.shift(2).fillna(False).astype(float)        # t 收盤訊號 → t+1 收盤成交 → t+2 起承擔報酬
    tr = hold.diff().abs().fillna(0)
    return hold * r + (1 - hold) * CASH - tr * COST, hold
def summ(ret):
    w = (1 + ret).cumprod(); return w.iloc[-1] ** (252 / len(ret)) - 1, (w / w.cummax() - 1).min()
def circ_p(hold, r, ret):
    hv, rv = hold.values, r.values; tot = np.prod(1 + ret.values); sims = []
    for k in range(60, len(hv) - 60, 3):
        hh = np.roll(hv, k); trn = np.abs(np.diff(np.r_[hh[0], hh]))
        sims.append(np.prod(1 + hh * rv + (1 - hh) * CASH - trn * COST))
    return (np.sum(np.array(sims) >= tot) + 1) / (len(sims) + 1)

L = []
def P(*a): s = " ".join(map(str, a)); print(s); L.append(s)
P(f"期間 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 交易日）")
for nm, r in (("0050", r50), ("正2", r2)):
    bh = summ(r); P(f"\n##### {nm}：一直持有 年化 {bh[0]:.2%}、最大回撤 {bh[1]:.1%}")
    rows = []
    for k, s in RULES.items():
        ret, hold = run(s, r); cg, dd = summ(ret)
        p = circ_p(hold, r, ret)
        n = int((hold.diff() > 0).sum())
        ok = cg > bh[0] and dd >= bh[1] and p < 0.05 / 9
        rows.append(dict(規則=k, 年化=f"{cg:.2%}", 最大回撤=f"{dd:.1%}", 持有比例=f"{hold.mean():.0%}", 進場次數=n,
                         勝過持有="✔" if cg > bh[0] else "", 回撤較小="✔" if dd > bh[1] else "", 隨機對照p=f"{p:.3f}", 通過="✔" if ok else ""))
    P(pd.DataFrame(rows).to_string(index=False))
# 各規則在主要空頭的表現（0050）
P("\n##### 主要空頭期間報酬（0050）")
eps = {"2008金融海嘯": ("2007-10-30", "2008-11-20"), "2011歐債": ("2011-02-08", "2011-12-19"), "2015": ("2015-04-27", "2015-08-24"),
       "2018": ("2018-01-23", "2018-10-26"), "2020疫情": ("2020-01-14", "2020-03-19"), "2022升息": ("2022-01-05", "2022-10-25"), "2025關稅": ("2025-01-21", "2025-04-09")}
rows = []
for k, s in {"持有": pd.Series(True, index=dates), **RULES}.items():
    ret, _ = run(s, r50); rows.append({"規則": k, **{e: f"{(1+ret.loc[a:b]).prod()-1:+.0%}" for e, (a, b) in eps.items()}})
P(pd.DataFrame(rows).to_string(index=False))
(OUT / "result.txt").write_text("\n".join(L), encoding="utf-8")
