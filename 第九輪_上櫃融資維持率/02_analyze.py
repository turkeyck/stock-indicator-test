"""計算上櫃融資維持率，並依 00_預先登記.md 回測"""
import json, gzip
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; RAW = HERE / "raw"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
# ---- 1) 維持率 ----
rows, last = [], {}
for f in sorted((RAW / "daily").glob("*.json.gz")):
    d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    cl = {k: v for k, v in d["close"].items() if v and v > 0}
    if not cl: continue                                   # 尚無收盤行情的早期日期
    last.update(cl)
    mv = cov = tot = 0.0
    for k, sh in d["margin"].items():
        if not sh: continue
        tot += sh
        if k in last: mv += sh * 1000 * last[k]; cov += sh
    if not d["loan_k"]: continue
    rows.append((pd.Timestamp(f.name[:8]), mv / (d["loan_k"] * 1000) * 100, cov / tot if tot else np.nan, d["loan_k"] / 1e5))
mr = pd.DataFrame(rows, columns=["date", "ratio", "coverage", "loan_億"]).set_index("date")
mr.to_csv(OUT / "otc_margin_ratio.csv", encoding="utf-8-sig")
# ---- 2) 價格 ----
def yahoo(fn):
    j = json.load(open(RAW / fn))["chart"]["result"][0]
    s = pd.Series(j["indicators"]["adjclose"][0]["adjclose"], index=pd.to_datetime(j["timestamp"], unit="s").normalize() + pd.Timedelta(hours=0)).dropna()
    s.index = (s.index + pd.Timedelta(hours=8)).normalize(); return s[~s.index.duplicated(keep="last")]
tx = pd.DataFrame(json.load(open(HERE.parent / "raw" / "price" / "taiex.json"))["data"])
taiex = pd.Series(tx.close.values, index=pd.to_datetime(tx.date)).astype(float)
txr = taiex.pct_change() + 0.03 / 252                         # 加權指數 + 3% 股息
e50 = yahoo("0050.TW.json"); l2 = yahoo("00631L.TW.json")
dates = mr.index
def splice(real, proxy_ret):
    r_real = real.pct_change()
    ix = proxy_ret.index.union(r_real.index)
    r = proxy_ret.reindex(ix); rr = r_real.reindex(ix); m = rr.notna(); m.iloc[:list(ix).index(r_real.index[1])] = False
    r[m] = rr[m]
    # 先換算成價格再對齊交易日（避免非交易日報酬遺失）
    w = (1 + r.fillna(0)).cumprod().reindex(ix.union(dates)).ffill().reindex(dates)
    return w.pct_change().fillna(0)
r50 = splice(e50, txr)
r2 = splice(l2, 2 * taiex.pct_change() + 0.03 / 252 * 2 - 0.01 / 252 - 0.01 / 252)   # 正2 模擬：2倍指數+股息 − 費用 − 借款
CASH = 0.01 / 252; COST = 0.001

def backtest(buy, sell, r, ratio=mr.ratio):
    pos = np.zeros(len(dates)); state = 0; sig = ratio.values
    for t in range(1, len(dates)):
        pos[t] = state                                         # t 日報酬由 t-1 日收盤後的部位決定
        if state == 0 and sig[t - 1] < buy: state = 1          # t-1 收盤判斷 → t 收盤成交
        elif state == 1 and sig[t - 1] > sell: state = 0
        pos[t] = pos[t]
    pos = pd.Series(pos, index=dates)
    # pos[t]=1 表示 t 日持有（於 t-1 收盤已持有）；交易於 t 收盤 → 從 t+1 起持有
    hold = pos.shift(1).fillna(0)
    trade = hold.diff().abs().fillna(0)
    ret = hold * r + (1 - hold) * CASH - trade * COST
    return ret, hold

def trades(hold, r):
    out = []; h = hold.values; rv = r.values; i = 0
    while i < len(h):
        if h[i] == 1:
            j = i
            while j < len(h) and h[j] == 1: j += 1
            seg = rv[i:j]; w = np.cumprod(1 + seg)
            out.append(dict(買進=str(dates[i - 1].date()), 賣出=str(dates[j - 1].date()) if j < len(h) else "未平倉",
                            天數=j - i, 報酬=w[-1] * (1 - COST) ** 2 - 1, 持有中最大回撤=(w / np.maximum.accumulate(np.r_[1, w])[1:] - 1).min()))
            i = j
        else: i += 1
    return pd.DataFrame(out)

def summ(ret):
    w = (1 + ret).cumprod(); yrs = len(ret) / 252
    return w.iloc[-1] ** (1 / yrs) - 1, (w / w.cummax() - 1).min()

L = []
def P(*a): s = " ".join(map(str, a)); print(s); L.append(s)
P(f"維持率期間：{dates[0].date()} ~ {dates[-1].date()}，{len(dates)} 個交易日；平均涵蓋率 {mr.coverage.mean():.1%}")
P(f"維持率分布：最低 {mr.ratio.min():.0f}（{mr.ratio.idxmin().date()}），最高 {mr.ratio.max():.0f}（{mr.ratio.idxmax().date()}），中位數 {mr.ratio.median():.0f}")
P(f"低於150的天數 {int((mr.ratio<150).sum())}，低於145 {int((mr.ratio<145).sum())}，高於205 {int((mr.ratio>205).sum())}，高於210 {int((mr.ratio>210).sum())}")
bh50 = summ(r50); bh2 = summ(r2)
P(f"\n買進持有 0050：年化 {bh50[0]:.2%}，最大回撤 {bh50[1]:.1%}｜買進持有 正2：年化 {bh2[0]:.2%}，最大回撤 {bh2[1]:.1%}")
res = []
for nm, r in (("0050", r50), ("正2", r2)):
    for b in (150, 145):
        for s in (210, 205):
            ret, hold = backtest(b, s, r); cg, dd = summ(ret); tb = trades(hold, r)
            # 循環平移對照：保留持有天數與交易結構，隨機改變時點
            hv = hold.values; rv = r.values; tot = np.prod(1 + ret.values)
            sims = []
            for k in range(60, len(hv) - 60):
                hh = np.roll(hv, k); tr_ = np.abs(np.diff(np.r_[0, hh]))
                sims.append(np.prod(1 + hh * rv + (1 - hh) * CASH - tr_ * COST))
            p = (np.sum(np.array(sims) >= tot) + 1) / (len(sims) + 1)
            P(f"\n=== {nm}｜<{b} 買、>{s} 賣 ===")
            P(f"年化 {cg:.2%}、最大回撤 {dd:.1%}、持有時間 {hold.mean():.0%}、交易 {len(tb)} 趟；隨機時點對照單尾 p = {p:.3f}")
            if len(tb):
                t2 = tb.copy(); t2["報酬"] = t2["報酬"].map(lambda v: f"{v:+.1%}"); t2["持有中最大回撤"] = t2["持有中最大回撤"].map(lambda v: f"{v:.1%}")
                P(t2.to_string(index=False))
                done = tb[tb["賣出"] != "未平倉"]
                P(f"已平倉 {len(done)} 趟：≥20% 的有 {(done['報酬']>=0.2).sum()} 趟，虧損 {(done['報酬']<0).sum()} 趟")
            res.append(dict(標的=nm, 買=b, 賣=s, 年化=cg, 回撤=dd, 趟數=len(tb), p=p))
pd.DataFrame(res).to_csv(OUT / "main_results.csv", index=False, encoding="utf-8-sig")
# ---- 敏感度 ----
grid = []
for b in range(125, 175, 5):
    for s in range(180, 245, 5):
        ret, hold = backtest(b, s, r50); cg, dd = summ(ret); tb = trades(hold, r50)
        done = tb[tb["賣出"] != "未平倉"] if len(tb) else tb
        grid.append(dict(買=b, 賣=s, 年化=cg, 回撤=dd, 趟數=len(tb), 已平倉=len(done), 全部賺20=bool(len(done) and (done["報酬"] >= .2).all())))
G = pd.DataFrame(grid); G.to_csv(OUT / "sensitivity_0050.csv", index=False, encoding="utf-8-sig")
P("\n=== 門檻敏感度（0050，年化報酬；持有 0050 = %.2f%%）===" % (bh50[0] * 100))
P(G.pivot(index="買", columns="賣", values="年化").map(lambda v: f"{v*100:.1f}").to_string())
P("\n已平倉交易「全部 ≥20%」的門檻組合：")
P(G.pivot(index="買", columns="賣", values="全部賺20").map(lambda v: "✔" if v else "·").to_string())
(OUT / "result.txt").write_text("\n".join(L), encoding="utf-8")
