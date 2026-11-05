"""個股因子組合搜尋：單一 + 兩兩 + 三組抽樣，× {不擇時, 10月均線濾網}；訓練挑選、驗證驗證、SPA"""
import itertools, time
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "out"
rng = np.random.default_rng(20261105)
Bk = pd.read_csv(OUT / "blocks.csv", index_col=0, encoding="utf-8-sig"); Bk.index = pd.PeriodIndex(Bk.index, freq="M")
Bk = Bk.loc["1963-07":"2026-09"]
mkt_all, rf_all = Bk.pop("_MKT").values, Bk.pop("_RF").values
names = list(Bk.columns); A = Bk.values.astype(np.float64); nB = len(names)
W0 = 6                                   # 1963-07~12 只作均線暖身
idx = Bk.index[W0:]; T = len(idx)
tr = np.asarray(idx <= pd.Period("1993-12", "M")); ho = ~tr; al = np.ones(T, bool)
mkt, rf = mkt_all[W0:], rf_all[W0:]

def combos():
    for i in range(nB): yield (i,)
    for c in itertools.combinations(range(nB), 2): yield c
    seen = set()
    while len(seen) < 150000:
        c = tuple(sorted(rng.choice(nB, 3, replace=False)))
        if c not in seen: seen.add(c); yield c

def returns(cs):
    """cs: list of tuples → (T×K 不擇時, T×K 均線濾網)"""
    K = len(cs); R = np.zeros((A.shape[0], K))
    for k, c in enumerate(cs): R[:, k] = A[:, list(c)].mean(1)
    w = np.cumprod(1 + R, axis=0); cs_ = np.cumsum(w, axis=0)
    ma = np.full_like(w, np.nan); ma[9:] = (cs_[9:] - np.vstack([np.zeros((1, K)), cs_[:-10]])) / 10
    on = np.zeros_like(w, bool); on[1:] = (w[:-1] > ma[:-1])          # 上月底在均線上 → 本月持有
    on[:10] = True
    Rf = np.where(on, R, rf_all[:, None])
    return R[W0:], Rf[W0:]

def metrics(Rm, b):
    n = Rm.shape[0]; w = np.cumprod(1 + Rm, axis=0)
    cg = w[-1] ** (12 / n) - 1; dd = (w / np.maximum.accumulate(w, axis=0) - 1).min(0)
    d = Rm - b[:, None]; t = d.mean(0) / (d.std(0, ddof=1) / np.sqrt(n) + 1e-12)
    return cg, dd, t
def stat_boot_idx(n, B, mb=12):
    out = np.empty((B, n), np.int32)
    for bb in range(B):
        i = rng.integers(n); jumps = rng.random(n) < 1 / mb; news = rng.integers(n, size=n)
        for t in range(n): out[bb, t] = i; i = news[t] if jumps[t] else (i + 1) % n
    return out
NB = 1000; masks = {"訓練": tr, "驗證": ho, "全期": al}
boots = {l: np.array([np.bincount(r_, minlength=m.sum()) for r_ in stat_boot_idx(m.sum(), NB)], np.float32) for l, m in masks.items()}
spa = {l: dict(T=0.0, Tb=np.zeros(NB, np.float32)) for l in masks}

rows = []; buf = []; CH = 5000; t0 = time.time(); nK = 0
def flush(buf):
    global nK
    R0, R1 = returns(buf)
    for tag, Rm in (("不擇時", R0), ("均線濾網", R1)):
        out = {"組合": [" + ".join(names[i] for i in c) for c in buf], "擇時": tag, "n": [len(c) for c in buf]}
        for l, m in masks.items():
            cg, dd, t = metrics(Rm[m], mkt[m]); out[f"{l}年化"] = cg; out[f"{l}回撤"] = dd; out[f"{l}t"] = t
            D = (Rm[m] - mkt[m][:, None]).astype(np.float32); n = D.shape[0]; db = D.mean(0); om = D.std(0, ddof=1) + 1e-9
            s = spa[l]; s["T"] = max(s["T"], float(np.max(np.sqrt(n) * db / om)))
            g = db * (np.sqrt(n) * db / om >= -np.sqrt(2 * np.log(np.log(n))))
            s["Tb"] = np.maximum(s["Tb"], np.max(np.sqrt(n) * (boots[l] @ D / n - g) / om, axis=1))
        rows.append(pd.DataFrame(out))
    nK += 2 * len(buf)
for c in combos():
    buf.append(c)
    if len(buf) == CH: flush(buf); buf = []; print(f"{nK:,} 個策略 {time.time()-t0:.0f}s", flush=True)
if buf: flush(buf)
res = pd.concat(rows, ignore_index=True)
for l, m in masks.items():
    w = np.cumprod(1 + mkt[m]); res[f"{l}市場年化"] = w[-1] ** (12 / m.sum()) - 1; res[f"{l}市場回撤"] = (w / np.maximum.accumulate(w) - 1).min()
res.to_csv(OUT / "all_strategies.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")

L = []
def P(*a): s = " ".join(map(str, a)); print(s); L.append(s)
P(f"\n策略總數：{len(res):,}（組合 {len(res)//2:,} × 擇時 2）")
for l in masks: P(f"{l}：市場 年化 {res[f'{l}市場年化'][0]:.2%} 回撤 {res[f'{l}市場回撤'][0]:.1%}")
ok = res[(res["訓練年化"] > res["訓練市場年化"]) & (res["訓練回撤"] >= res["訓練市場回撤"])]
P(f"訓練期「報酬>市場且回撤≤市場」：{len(ok):,} 個")
top = ok.sort_values("訓練t", ascending=False).head(25); best = top.index[0]
cs = tuple(names.index(x) for x in res.loc[best, "組合"].split(" + "))
R0, R1 = returns([cs]); rb = (R0 if res.loc[best, "擇時"] == "不擇時" else R1)[:, 0]
d = (rb - mkt)[ho]; I = stat_boot_idx(len(d), 5000); p_ho = ((d[I].mean(1) - d.mean()) >= d.mean()).mean()
P("\n=== 主要：訓練期 t 值最高者 ===")
P(res.loc[best, "組合"], "｜", res.loc[best, "擇時"])
P(f"訓練：年化 {res.loc[best,'訓練年化']:.2%} 回撤 {res.loc[best,'訓練回撤']:.1%} t={res.loc[best,'訓練t']:.2f}")
P(f"驗證：年化 {res.loc[best,'驗證年化']:.2%}（市場 {res.loc[best,'驗證市場年化']:.2%}） 回撤 {res.loc[best,'驗證回撤']:.1%}（市場 {res.loc[best,'驗證市場回撤']:.1%}） 超額單尾p={p_ho:.4f}")
P("\n=== 訓練期前 25 名在驗證期 ===")
sh = top[["組合", "擇時", "訓練年化", "訓練回撤", "訓練t", "驗證年化", "驗證回撤", "驗證t"]].copy()
for c in sh.columns[2:]: sh[c] = sh[c].map(lambda v: f"{v:.2%}" if "年化" in c else (f"{v:.1%}" if "回撤" in c else f"{v:.2f}"))
P(sh.to_string(index=False))
win = ok[(ok["驗證年化"] > ok["驗證市場年化"]) & (ok["驗證回撤"] >= ok["驗證市場回撤"])]
P(f"\n訓練合格者中，驗證期也「報酬>市場且回撤≤市場」：{len(win):,} / {len(ok):,}（{len(win)/max(len(ok),1):.1%}）")
for l in masks: s = spa[l]; P(f"SPA（{len(res):,} 個策略，{l}）：最大 t={s['T']:.2f}，校正後 p={(s['Tb'] >= s['T']).mean():.4f}")
(OUT / "search_result.txt").write_text("\n".join(L), encoding="utf-8")
