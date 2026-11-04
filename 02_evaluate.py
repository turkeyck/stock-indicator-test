"""第四輪檢定（依 00_預先登記.md）"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent; RAW = HERE / "raw"; OUT = HERE / "out"
rng = np.random.default_rng(20261104)

sc = pd.read_csv(OUT / "score_monthly.csv", index_col=0, encoding="utf-8-sig")
sc.index = pd.PeriodIndex(sc.index, freq="M")

# ---- 價格 ----
lead = pd.read_csv(RAW / "ndc" / "領先指標構成項目.csv", na_values=["-"])
avgpx = pd.Series(lead["股價指數(Index1966=100)"].values, index=pd.PeriodIndex(lead.Date.astype(str), freq="M")).astype(float)
comp = pd.read_csv(RAW / "ndc" / "景氣指標與燈號.csv", na_values=["-"])
coin = pd.Series(comp["同時指標不含趨勢指數"].values, index=pd.PeriodIndex(comp.Date.astype(str), freq="M")).astype(float)
j = json.load(open(RAW / "price" / "twii.json"))["chart"]["result"][0]
d = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s")).dropna()
mepx = d.resample("ME").last(); mepx.index = mepx.index.to_period("M")

DIV, CASH, BORROW = 0.03 / 12, 0.01 / 12, 0.02 / 12

def nw_t(y, x, L=12):
    """OLS 斜率的 Newey-West t 值"""
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1); G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None]); S += w * (G + G.T)
    V = XtXi @ S @ XtXi; return b[1], b[1] / np.sqrt(V[1, 1])

def circ_p(y, x, nsim=5000):
    """循環平移隨機化：保留兩序列各自的自相關，打亂相對時間，單尾 p"""
    obs = np.corrcoef(x, y)[0, 1]; n = len(x); cnt = 0
    for _ in range(nsim):
        k = rng.integers(12, n - 12); cnt += np.corrcoef(np.roll(x, k), y)[0, 1] >= obs
    return obs, (cnt + 1) / (nsim + 1)

def setup(px, extra_lag):
    """回傳對齊後的 DataFrame。extra_lag=1：月平均價格時訊號再延後 1 個月"""
    df = pd.DataFrame({"px": px}).join(sc[["composite", "level"]].shift(extra_lag))
    r = df.px.pct_change()
    df["ret_next"] = r.shift(-1)                                  # 決策月之後一個月的報酬
    df["fwd12"] = np.log(df.px.shift(-12) / df.px)                  # 未來 12 個月
    df["fwd6"] = np.log(df.px.shift(-6) / df.px)
    return df

def bears(px):
    """找出 ≥20% 空頭；回傳 (高點, 低點, 月數, 跌幅)"""
    out = []; peak = px.iloc[0]; pk = px.index[0]; trough = peak; tr = pk; inbear = False
    for t, v in px.items():
        if not inbear:
            if v > peak: peak, pk = v, t
            elif v / peak - 1 <= -0.20: inbear, trough, tr = True, v, t
        else:
            if v < trough: trough, tr = v, t
            elif v / trough - 1 >= 0.20:
                out.append((pk, tr, (tr - pk).n, trough / peak - 1)); inbear = False; peak, pk = v, t
    if inbear: out.append((pk, tr, (tr - pk).n, trough / peak - 1))
    return out

def backtest(df, rule):
    e = df.level.map(rule)
    r = e * (df.ret_next + DIV) + (1 - e).clip(lower=0) * CASH - (e - 1).clip(lower=0) * BORROW
    return r.dropna()

def stats_of(r):
    w = (1 + r).cumprod(); yrs = len(r) / 12
    return dict(CAGR=w.iloc[-1] ** (1 / yrs) - 1, MDD=(w / w.cummax() - 1).min(), vol=r.std() * np.sqrt(12))

RULE_A = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.0, 5: 1.0}
RULE_B = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.3, 5: 1.3}
BH = {k: 1.0 for k in range(1, 6)}

lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

for name, px, xl, start in [("主樣本：國發會月平均指數（訊號多延後1月）", avgpx, 1, "1992-01"),
                            ("穩健性：每日月底收盤（1997後）", mepx, 0, "1998-01")]:
    df = setup(px, xl).loc[start:]
    P(f"\n================ {name} ================")
    P("樣本:", df.composite.first_valid_index(), "~", df.index[-1])
    # H1
    z = df[["composite", "fwd12"]].dropna()
    b, t = nw_t(z.fwd12.values, z.composite.values); rho, p = circ_p(z.fwd12.values, z.composite.values)
    P(f"H1 綜合分數→未來12月報酬：斜率={b:.3f}, NW t={t:.2f} (單尾p={1-stats.norm.cdf(t):.4f}), corr={rho:.3f}, 循環平移p={p:.4f}, N={len(z)}")
    z6 = df[["composite", "fwd6"]].dropna(); b6, t6 = nw_t(z6.fwd6.values, z6.composite.values, 6)
    P(f"   （參考）未來6月：斜率={b6:.3f}, NW t={t6:.2f}")
    for a, bb in [(start, "2008-12"), ("2009-01", "2026-12")]:
        zz = z.loc[a:bb]; bh, th = nw_t(zz.fwd12.values, zz.composite.values)
        P(f"   子樣本 {a}~{bb}: 斜率={bh:.3f}, NW t={th:.2f}, N={len(zz)}")
    # H2
    g = df.dropna(subset=["level", "fwd12"]).groupby("level")
    tab = pd.DataFrame({"月數": g.size(), "未來12月平均報酬": g.fwd12.mean().map(lambda v: f"{np.expm1(v):.1%}"),
                        "未來12月下跌機率": g.fwd12.apply(lambda v: f"{(v < 0).mean():.0%}"),
                        "未來12月跌>15%機率": g.fwd12.apply(lambda v: f"{(v < np.log(0.85)).mean():.0%}"),
                        "下個月平均報酬": g.ret_next.mean().map(lambda v: f"{v:.2%}")})
    P(tab.to_string())
    zz = df.dropna(subset=["level", "fwd12"])
    P(f"H2 等級與未來12月報酬 Spearman={stats.spearmanr(zz.level, zz.fwd12)[0]:.3f}")
    l1 = zz.level == 1; diff = zz.fwd12[~l1].mean() - zz.fwd12[l1].mean()
    # 區塊自助法：L1 與其他差異（以循環平移等級序列）
    sims = []
    lv = zz.level.values; f = zz.fwd12.values
    for _ in range(5000):
        k = rng.integers(12, len(lv) - 12); m = np.roll(lv, k) == 1
        sims.append(f[~m].mean() - f[m].mean())
    P(f"   L1 vs 其他 未來12月 log 報酬差={diff:.3f}, 循環平移單尾p={(np.sum(np.array(sims) >= diff) + 1) / 5001:.4f}")
    # H3：景氣本身
    if xl == 1:
        h3 = pd.DataFrame({"c": sc.composite, "dc": coin.shift(-6) - coin}).dropna().loc[start:]
        b3, t3 = nw_t(h3.dc.values, h3.c.values, 6); r3, p3 = circ_p(h3.dc.values, h3.c.values)
        P(f"H3 綜合分數→同時指標(不含趨勢)未來6月變化：NW t={t3:.2f}, corr={r3:.3f}, 循環平移p={p3:.4f}")
    # H4
    for nm, rule in [("買進持有", BH), ("規則A(L1=0,L2=50%)", RULE_A), ("規則B(+L4/5 130%)", RULE_B)]:
        r = backtest(df, rule); s = stats_of(r)
        P(f"H4 {nm:22s} 年化={s['CAGR']:.2%} 最大回撤={s['MDD']:.1%} 波動={s['vol']:.1%} 平均曝險={df.level.map(rule).mean():.2f}")
    # 空頭分類
    P("空頭（≥20%）：高點→低點, 月數, 跌幅, 類型, 高點前後的等級")
    for pk, tr, n, dd in bears(px.loc["1990-01":]):
        typ = "急跌" if n <= 4 else "緩跌"
        lv = df.level.loc[pk - 3: min(tr, df.index[-1])].dropna().astype(int)
        seq = "".join(map(str, lv.values))
        P(f"  {pk}→{tr} {n:3d}月 {dd:.0%} {typ}  等級(高點前3月起):{seq}")

(OUT / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")
