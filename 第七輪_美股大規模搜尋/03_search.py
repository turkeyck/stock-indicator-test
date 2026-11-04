"""產生 >100,000 條規則 × 2 曝險，訓練期挑選、驗證期驗證、SPA 校正（分塊計算）"""
import itertools, time
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "out"
rng = np.random.default_rng(20261105)
X = pd.read_csv(OUT / "features.csv", index_col=0, encoding="utf-8-sig"); X.index = pd.PeriodIndex(X.index, freq="M")
sg = pd.read_csv(OUT / "theory_sign.csv", index_col=0, encoding="utf-8-sig").iloc[:, 0]
Rt = pd.read_csv(OUT / "returns.csv", index_col=0, encoding="utf-8-sig"); Rt.index = pd.PeriodIndex(Rt.index, freq="M")
idx = pd.period_range("1966-01", "2026-09", freq="M")
r = Rt.r.reindex(idx).values.astype(np.float64); cash = Rt.cash.reindex(idx).values.astype(np.float64); borrow = cash + 0.01 / 12
X = X.reindex(idx); feats = list(X.columns); P = X.values; nanm = np.isnan(P); nF = len(feats)
T = len(idx); tr = np.asarray(idx <= pd.Period("1995-12", "M")); ho = ~tr; al = np.ones(T, bool)
sgn = np.array([sg[f] for f in feats])
with np.errstate(invalid="ignore"):
    theo = {q: ((P - q) * sgn > 0) | nanm for q in (.3, .4, .5, .6, .7)}   # 指標不可得 → 視為有利（持有）
px_trend = X[[c for c in feats if c.startswith("T10_")][0]].values

def gen():
    """依序產生 (名稱, 有利布林向量)"""
    for j, f in enumerate(feats):
        for d in (1, -1):
            for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                with np.errstate(invalid="ignore"): s = ((P[:, j] > q) if d > 0 else (P[:, j] < q)) | nanm[:, j]
                yield f"單一|{f}|{'高' if d>0 else '低'}於{int(q*100)}", s
    for a, b in itertools.combinations(range(nF), 2):
        for q in (.3, .5, .7):
            yield f"AND{int(q*100)}|{feats[a]}|{feats[b]}", theo[q][:, a] & theo[q][:, b]
            yield f"OR{int(q*100)}|{feats[a]}|{feats[b]}", theo[q][:, a] | theo[q][:, b]
    for a, b, c in itertools.combinations(range(nF), 3):
        for q in (.3, .4, .5, .6, .7):
            t = theo[q]; yield f"多數決{int(q*100)}|{feats[a]}|{feats[b]}|{feats[c]}", (t[:, a].astype(np.int8) + t[:, b] + t[:, c]) >= 2
    with np.errstate(invalid="ignore"): trend = (px_trend > 0.5) | np.isnan(px_trend)
    for j, f in enumerate(feats):
        for d in (1, -1):
            for q in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
                with np.errstate(invalid="ignore"): s = ((P[:, j] > q) if d > 0 else (P[:, j] < q)) | nanm[:, j]
                yield f"趨勢AND|{f}|{'高' if d>0 else '低'}於{int(q*100)}", s & trend

bh = r
def metrics(Rm, b):
    n = Rm.shape[0]; w = np.cumprod(1 + Rm, axis=0)
    cg = w[-1] ** (12 / n) - 1; dd = (w / np.maximum.accumulate(w, axis=0) - 1).min(0)
    dlt = Rm - b[:, None]; t = dlt.mean(0) / (dlt.std(0, ddof=1) / np.sqrt(n) + 1e-12)
    return cg, dd, t

def stat_boot_idx(n, B, mb=12):
    out = np.empty((B, n), np.int32)
    for bb in range(B):
        i = rng.integers(n); jumps = rng.random(n) < 1 / mb; news = rng.integers(n, size=n)
        for t in range(n):
            out[bb, t] = i; i = news[t] if jumps[t] else (i + 1) % n
    return out
B = 1000
boots = {lab: np.array([np.bincount(row, minlength=m.sum()) for row in stat_boot_idx(m.sum(), B)], dtype=np.float32) for lab, m in (("訓練", tr), ("驗證", ho), ("全期", al))}
spa_state = {lab: dict(Tobs=0.0, Tb=np.zeros(B, np.float32)) for lab in boots}
masks = {"訓練": tr, "驗證": ho, "全期": al}

rows = []; names = []; CH = 4000; buf = []; t0 = time.time(); K = 0
def flush(buf):
    global K
    nm = [n for n, _ in buf]; FAV = np.array([s for _, s in buf], dtype=np.float32).T
    for L in (1.0, 1.5):
        e = FAV * L
        Rm = e * r[:, None] + (1 - e).clip(min=0) * cash[:, None] - (e - 1).clip(min=0) * borrow[:, None]
        out = {"策略": [f"{n}|曝險{int(L*100)}%" for n in nm], "平均曝險": e.mean(0)}
        for lab, m in masks.items():
            cg, dd, t = metrics(Rm[m], bh[m]); out[f"{lab}年化"] = cg; out[f"{lab}回撤"] = dd; out[f"{lab}t"] = t
            Dm = (Rm[m] - bh[m][:, None]).astype(np.float32); n = Dm.shape[0]
            db = Dm.mean(0); om = Dm.std(0, ddof=1) + 1e-9
            st = spa_state[lab]; st["Tobs"] = max(st["Tobs"], float(np.max(np.sqrt(n) * db / om)))
            g = db * (np.sqrt(n) * db / om >= -np.sqrt(2 * np.log(np.log(n))))
            Db = boots[lab] @ Dm / n
            st["Tb"] = np.maximum(st["Tb"], np.max(np.sqrt(n) * (Db - g) / om, axis=1))
        rows.append(pd.DataFrame(out)); K += len(nm)
for item in gen():
    buf.append(item)
    if len(buf) == CH: flush(buf); buf = []; print(f"{K*2:,} 個策略 {time.time()-t0:.0f}s", flush=True)
if buf: flush(buf)
res = pd.concat(rows, ignore_index=True)
for lab, m in masks.items():
    w = np.cumprod(1 + bh[m]); res[f"{lab}持有年化"] = w[-1] ** (12 / m.sum()) - 1; res[f"{lab}持有回撤"] = (w / np.maximum.accumulate(w) - 1).min()
    e = 1.5; x = e * bh[m] - .5 * borrow[m]; w = np.cumprod(1 + x); res[f"{lab}150%持有年化"] = w[-1] ** (12 / m.sum()) - 1
res.to_parquet(OUT / "all_strategies.parquet") if False else res.to_csv(OUT / "all_strategies.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

lines = []
def Pr(*a): s = " ".join(map(str, a)); print(s); lines.append(s)
Pr(f"\n策略總數：{len(res):,}（規則 {len(res)//2:,} × 曝險 2）")
for lab in masks: Pr(f"{lab}：持有 年化 {res[f'{lab}持有年化'][0]:.2%} 回撤 {res[f'{lab}持有回撤'][0]:.1%}；固定150% 年化 {res[f'{lab}150%持有年化'][0]:.2%}")
ok = res[(res["訓練年化"] > res["訓練持有年化"]) & (res["訓練回撤"] >= res["訓練持有回撤"])]
Pr(f"訓練期「報酬>持有且回撤≤持有」：{len(ok):,} 個")
top = ok.sort_values("訓練t", ascending=False).head(25); best = top.index[0]
# 驗證期 bootstrap p（重算該策略報酬）
def strat_returns(name):
    for n, s in gen():
        if name.startswith(n + "|曝險"):
            L = 1.5 if name.endswith("150%") else 1.0; e = s.astype(float) * L
            return e * r + (1 - e).clip(min=0) * cash - (e - 1).clip(min=0) * borrow
rb = strat_returns(res.loc[best, "策略"]); d = (rb - bh)[ho]
I = stat_boot_idx(len(d), 5000); p_ho = ((d[I].mean(1) - d.mean()) >= d.mean()).mean()
Pr("\n=== 主要：訓練期 t 值最高者 ===")
Pr(res.loc[best, "策略"], f"平均曝險 {res.loc[best,'平均曝險']:.2f}")
Pr(f"訓練：年化 {res.loc[best,'訓練年化']:.2%} 回撤 {res.loc[best,'訓練回撤']:.1%} t={res.loc[best,'訓練t']:.2f}")
Pr(f"驗證：年化 {res.loc[best,'驗證年化']:.2%}（持有 {res.loc[best,'驗證持有年化']:.2%}，固定150% {res.loc[best,'驗證150%持有年化']:.2%}） 回撤 {res.loc[best,'驗證回撤']:.1%}（持有 {res.loc[best,'驗證持有回撤']:.1%}） 超額單尾p={p_ho:.4f}")
Pr("\n=== 訓練期前 25 名在驗證期 ===")
sh = top[["策略", "訓練年化", "訓練回撤", "訓練t", "驗證年化", "驗證回撤", "驗證t"]].copy()
for c in sh.columns[1:]: sh[c] = sh[c].map(lambda v: f"{v:.2%}" if "年化" in c else (f"{v:.1%}" if "回撤" in c else f"{v:.2f}"))
Pr(sh.to_string(index=False))
win = ok[(ok["驗證年化"] > ok["驗證持有年化"]) & (ok["驗證回撤"] >= ok["驗證持有回撤"])]
Pr(f"\n訓練合格者中，驗證期也「報酬>持有且回撤≤持有」：{len(win):,} / {len(ok):,}（{len(win)/max(len(ok),1):.1%}）")
win2 = win[win["驗證年化"] > win["驗證150%持有年化"]]; Pr(f"其中驗證期還勝過固定150%：{len(win2):,}")
for lab in masks:
    st = spa_state[lab]; Pr(f"SPA（{len(res):,} 個策略，{lab}）：最大 t={st['Tobs']:.2f}，校正後 p={(st['Tb'] >= st['Tobs']).mean():.4f}")
(OUT / "search_result.txt").write_text("\n".join(lines), encoding="utf-8")
