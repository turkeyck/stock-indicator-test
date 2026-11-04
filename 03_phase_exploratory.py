"""第二階段（探索性，看過 02 結果後才提出 → 須用更嚴格門檻並靠前瞻驗證）
經典「景氣四階段」：綜合分數的 水準(高/低) × 方向(上升/下降)
  復甦=低且上升、擴張=高且上升、趨緩=高且下降、衰退=低且下降
參數只試兩組方向窗口（3 月、6 月），Bonferroni：α = 0.05/2 = 0.025
"""
import importlib.util, sys
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ev", HERE / "02_evaluate.py")
src = (HERE / "02_evaluate.py").read_text(encoding="utf-8").split("lines = []")[0]
ns = {"__file__": str(HERE / "02_evaluate.py")}; exec(src, ns)
setup, nw_t, circ_p, stats_of, sc, avgpx, mepx, DIV, CASH, BORROW, bears = (ns[k] for k in
    ["setup", "nw_t", "circ_p", "stats_of", "sc", "avgpx", "mepx", "DIV", "CASH", "BORROW", "bears"])
rng = np.random.default_rng(7)
PH = {1: "趨緩(高↓)", 2: "衰退(低↓)", 3: "擴張(高↑)", 4: "復甦(低↑)"}
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

def phase(c, w):
    d = c - c.shift(w); hi = c > c.expanding(36).median()     # 高/低用自身歷史中位數（無未來資訊）
    ph = np.select([hi & (d < 0), ~hi & (d < 0), hi & (d >= 0), ~hi & (d >= 0)], [1, 2, 3, 4], np.nan)
    return pd.Series(np.where(c.notna() & d.notna(), ph, np.nan), index=c.index), d

for w in (3, 6):
    for name, px, xl, start in [("月平均指數(多延後1月) 1992-", avgpx, 1, "1992-01"), ("月底收盤 1998-", mepx, 0, "1998-01")]:
        ph, dchg = phase(sc.composite, w)
        df = setup(px, xl).join(pd.DataFrame({"ph": ph, "dchg": dchg}).shift(xl)).loc[start:]
        P(f"\n===== 方向窗口 {w} 月｜{name} =====")
        z = df.dropna(subset=["ph", "fwd12"]); g = z.groupby("ph")
        tab = pd.DataFrame({"月數": g.size(), "未來12月平均": g.fwd12.mean().map(lambda v: f"{np.expm1(v):.1%}"),
                            "未來12月跌>15%機率": g.fwd12.apply(lambda v: f"{(v < np.log(.85)).mean():.0%}"),
                            "下月平均": g.ret_next.mean().map(lambda v: f"{v:.2%}")}); tab.index = tab.index.map(lambda k: PH[int(k)])
        P(tab.to_string())
        # 主要探索檢定：分數「變化」→ 未來 12 月報酬
        zz = df[["dchg", "fwd12"]].dropna(); b, t = nw_t(zz.fwd12.values, zz.dchg.values); r, p = circ_p(zz.fwd12.values, zz.dchg.values)
        P(f"分數{w}月變化→未來12月：NW t={t:.2f}, corr={r:.3f}, 循環平移單尾p={p:.4f}  (門檻0.025)")
        zz = df[["dchg", "ret_next"]].dropna(); b, t = nw_t(zz.ret_next.values, zz.dchg.values, 3); r, p = circ_p(zz.ret_next.values, zz.dchg.values)
        P(f"分數{w}月變化→下個月  ：NW t={t:.2f}, corr={r:.3f}, 循環平移單尾p={p:.4f}")
        # 趨緩 vs 其他
        m = (z.ph == 1).values; f = z.fwd12.values; obs = f[~m].mean() - f[m].mean(); sims = []
        for _ in range(5000):
            k = rng.integers(12, len(m) - 12); mm = np.roll(m, k); sims.append(f[~mm].mean() - f[mm].mean())
        P(f"趨緩 vs 其他 未來12月 log 差={obs:.3f}, 循環平移單尾p={(np.sum(np.array(sims) >= obs) + 1) / 5001:.4f}")
        # 規則：趨緩=0%（或50%）、其他=100%；以及 衰退=50%
        for nm, rule in [("買進持有", {1: 1, 2: 1, 3: 1, 4: 1}), ("趨緩0%", {1: 0, 2: 1, 3: 1, 4: 1}),
                         ("趨緩0%/衰退50%", {1: 0, 2: .5, 3: 1, 4: 1}), ("趨緩50%/復甦130%", {1: .5, 2: 1, 3: 1, 4: 1.3})]:
            e = df.ph.map(rule)
            rr = (e * (df.ret_next + DIV) + (1 - e).clip(lower=0) * CASH - (e - 1).clip(lower=0) * BORROW).dropna()
            s = stats_of(rr); P(f"  {nm:18s} 年化={s['CAGR']:.2%} 最大回撤={s['MDD']:.1%} 平均曝險={e.mean():.2f}")
        if w == 6 and xl == 1:
            P("空頭期間階段序列（1趨緩 2衰退 3擴張 4復甦），高點前6月起：")
            for pk, tr, n, dd in bears(px.loc["1990-01":]):
                seq = "".join(map(str, df.ph.loc[pk - 6: min(tr, df.index[-1])].dropna().astype(int).values))
                P(f"  {pk}→{tr} {n}月 {dd:.0%}  {seq}")
(HERE / "out" / "phase_exploratory.txt").write_text("\n".join(lines), encoding="utf-8")
