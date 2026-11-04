"""產生 >10000 個策略，訓練期挑選、驗證期驗證、SPA 校正（依 00_預先登記.md）"""
import itertools
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "out"
rng = np.random.default_rng(20261105)
X = pd.read_csv(OUT / "features.csv", index_col=0, encoding="utf-8-sig"); X.index = pd.PeriodIndex(X.index, freq="M")
sg = pd.read_csv(OUT / "theory_sign.csv", index_col=0, encoding="utf-8-sig").iloc[:, 0]
px = pd.read_csv(OUT / "px.csv", index_col=0, encoding="utf-8-sig").iloc[:, 0]; px.index = pd.PeriodIndex(px.index, freq="M")
DIV, CASH, BORROW = .03 / 12, .01 / 12, .02 / 12
idx = pd.period_range("1998-01", "2026-09", freq="M")
rnext = (px.shift(-1) / px - 1).reindex(idx).values
X = X.reindex(idx); feats = list(X.columns); P = X.values           # T × F 百分位
nanm = np.isnan(P)
T = len(idx); tr = idx <= pd.Period("2012-12", "M"); ho = ~tr

names, favs = [], []
def add(name, fav, nanmask):
    f = fav.copy(); f[nanmask] = True                                 # 指標不可得 → 視為中性持有
    names.append(name); favs.append(f)
Q = [.2, .35, .5, .65, .8]
for j, f in enumerate(feats):                                         # 1) 單一門檻
    for d in (1, -1):
        for q in Q:
            add(f"單一|{f}|{'高' if d>0 else '低'}於{int(q*100)}", (P[:, j] > q) if d > 0 else (P[:, j] < q), nanm[:, j])
theo = np.array([(P[:, j] - .5) * sg[f] > 0 for j, f in enumerate(feats)]).T
for a, b in itertools.combinations(range(len(feats)), 2):              # 2) 兩兩組合
    m = nanm[:, a] | nanm[:, b]
    add(f"AND|{feats[a]}|{feats[b]}", theo[:, a] & theo[:, b], m)
    add(f"OR|{feats[a]}|{feats[b]}", theo[:, a] | theo[:, b], m)
jt = feats.index("P10_台股對10月均"); trend = P[:, jt] > np.nanpercentile(P[:, jt], 0) ; trend = (X["P10_台股對10月均"].values > 0.5)
# 趨勢濾網：用「台股 > 10 月均線」本身（原始比值>0）較直觀，百分位資料中取 >0.5 近似；另以原始值重算
raw10 = (px / px.rolling(10).mean() - 1).reindex(idx).values > 0
for j, f in enumerate(feats):                                         # 3) 趨勢濾網 + 單一
    if f.startswith("P10"): continue
    for d in (1, -1):
        for q in Q:
            s = (P[:, j] > q) if d > 0 else (P[:, j] < q); s = s | nanm[:, j]
            names.append(f"趨勢AND|{f}|{'高' if d>0 else '低'}於{int(q*100)}"); favs.append(raw10 & s)
FAV = np.array(favs).T.astype(float)                                  # T × K
K0 = FAV.shape[1]
R_list, nm_all = [], []
for L in (1.0, 1.5):
    e = FAV * L
    R_list.append(e * (rnext[:, None] + DIV) + (1 - e).clip(min=0) * CASH - (e - 1).clip(min=0) * BORROW)
    nm_all += [f"{n}|曝險{int(L*100)}%" for n in names]
R = np.hstack(R_list); K = R.shape[1]
bh = rnext + DIV
print("策略總數:", K)

def cagr(r): return np.prod(1 + r, axis=0) ** (12 / r.shape[0]) - 1
def mdd(r):
    w = np.cumprod(1 + r, axis=0); return (w / np.maximum.accumulate(w, axis=0) - 1).min(axis=0)
def tstat(d): return d.mean(0) / (d.std(0, ddof=1) / np.sqrt(d.shape[0]) + 1e-12)

res = pd.DataFrame({"策略": nm_all})
for lab, m in (("訓練", tr), ("驗證", ho), ("全期", np.ones(T, bool))):
    r = R[m]; b = bh[m]
    res[f"{lab}年化"] = cagr(r); res[f"{lab}回撤"] = mdd(r); res[f"{lab}t"] = tstat(r - b[:, None])
    res[f"{lab}持有年化"] = cagr(b[:, None])[0]; res[f"{lab}持有回撤"] = mdd(b[:, None])[0]
res.to_csv(OUT / "all_strategies.csv", index=False, encoding="utf-8-sig")

def stat_boot_idx(n, B, mean_block=12):
    p = 1 / mean_block; out = np.empty((B, n), int)
    for b in range(B):
        i = rng.integers(n)
        for t in range(n):
            out[b, t] = i; i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return out

def spa(D, B=2000):
    """Hansen SPA（consistent），D: n × K 超額報酬。回傳 p 值"""
    n = D.shape[0]; dbar = D.mean(0); om = D.std(0, ddof=1) + 1e-12
    Tobs = np.max(np.sqrt(n) * dbar / om); Tobs = max(Tobs, 0)
    g = dbar * (np.sqrt(n) * dbar / om >= -np.sqrt(2 * np.log(np.log(n))))
    I = stat_boot_idx(n, B); cnt = np.zeros((B, n))
    for b in range(B): cnt[b] = np.bincount(I[b], minlength=n)
    Db = cnt @ D / n                                                     # B × K
    Tb = np.max(np.sqrt(n) * (Db - g) / om, axis=1); Tb = np.maximum(Tb, 0)
    return (Tb >= Tobs).mean(), Tobs

def boot_p(d, B=5000):
    n = len(d); I = stat_boot_idx(n, B); m = d[I].mean(1) - d.mean(); return (m >= d.mean()).mean()

lines = []
def Pr(*a): s = " ".join(map(str, a)); print(s); lines.append(s)
Pr(f"策略總數 {K}（規則 {K0} × 曝險 2）")
Pr(f"買進持有：訓練 年化 {res['訓練持有年化'][0]:.2%} 回撤 {res['訓練持有回撤'][0]:.1%}｜驗證 年化 {res['驗證持有年化'][0]:.2%} 回撤 {res['驗證持有回撤'][0]:.1%}｜全期 年化 {res['全期持有年化'][0]:.2%} 回撤 {res['全期持有回撤'][0]:.1%}")
ok = res[(res["訓練年化"] > res["訓練持有年化"]) & (res["訓練回撤"] >= res["訓練持有回撤"])]
Pr(f"訓練期同時「報酬>持有、回撤≤持有」的策略：{len(ok)} 個")
top = ok.sort_values("訓練t", ascending=False).head(20)
best = top.index[0]
d_ho = R[ho, best] - bh[ho]
Pr("\n=== 主要：訓練期 t 值最高者 ===")
Pr(res.loc[best, "策略"])
Pr(f"訓練：年化 {res.loc[best,'訓練年化']:.2%} 回撤 {res.loc[best,'訓練回撤']:.1%} t={res.loc[best,'訓練t']:.2f}")
Pr(f"驗證：年化 {res.loc[best,'驗證年化']:.2%}（持有 {res.loc[best,'驗證持有年化']:.2%}） 回撤 {res.loc[best,'驗證回撤']:.1%}（持有 {res.loc[best,'驗證持有回撤']:.1%}） 超額單尾p={boot_p(d_ho):.4f}")
Pr("\n=== 訓練期前 20 名在驗證期的表現 ===")
show = top[["策略", "訓練年化", "訓練回撤", "訓練t", "驗證年化", "驗證回撤", "驗證t"]].copy()
for c in show.columns[1:]: show[c] = show[c].map(lambda v: f"{v:.2%}" if "年化" in c else (f"{v:.1%}" if "回撤" in c else f"{v:.2f}"))
Pr(show.to_string(index=False))
win = ok.index[(res.loc[ok.index, "驗證年化"] > res.loc[ok.index, "驗證持有年化"]) & (res.loc[ok.index, "驗證回撤"] >= res.loc[ok.index, "驗證持有回撤"])]
Pr(f"\n訓練期合格的 {len(ok)} 個中，驗證期也「報酬>持有且回撤≤持有」的有 {len(win)} 個（{len(win)/max(len(ok),1):.0%}）")
# SPA
for lab, m in (("訓練期", tr), ("驗證期", ho), ("全期", np.ones(T, bool))):
    p, t = spa(R[m] - bh[m][:, None]); Pr(f"SPA（{K} 個策略，{lab}）：最大 t={t:.2f}，校正後 p={p:.4f}")
# 全期最佳（純事後，僅供對照）
fb = res.sort_values("全期t", ascending=False).iloc[0]
Pr(f"\n（事後對照）全期 t 最高：{fb['策略']} 年化 {fb['全期年化']:.2%} 回撤 {fb['全期回撤']:.1%} t={fb['全期t']:.2f}")
(OUT / "search_result.txt").write_text("\n".join(lines), encoding="utf-8")
