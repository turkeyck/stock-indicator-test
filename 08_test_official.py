"""第五輪檢定：官方景氣燈號／領先指標（依 10_預先登記_官方燈號即時資料.md）"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
HERE = Path(__file__).parent; RAW = HERE / "raw"; OUT = HERE / "out"
ns = {"__file__": str(HERE / "02_evaluate.py")}
exec((HERE / "02_evaluate.py").read_text(encoding="utf-8").split("lines = []")[0], ns)
nw_t, circ_p, stats_of, mepx, avgpx, DIV, CASH = (ns[k] for k in ["nw_t", "circ_p", "stats_of", "mepx", "avgpx", "DIV", "CASH"])
rng = np.random.default_rng(11)
lines = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); lines.append(s)

# ---- 修正後（今天下載）官方資料 ----
c = pd.read_csv(RAW / "ndc" / "景氣指標與燈號.csv", na_values=["-"]); c.index = pd.PeriodIndex(c.Date.astype(str), freq="M")
score_rev = pd.to_numeric(c["景氣對策信號綜合分數"], errors="coerce")
lead_rev = pd.to_numeric(c["領先指標不含趨勢指數"], errors="coerce")
lead_dir_rev = np.sign(lead_rev.diff())
# ---- 當時公布（新聞）----
rt = pd.read_csv(OUT / "realtime_signals.csv"); rt.index = pd.PeriodIndex(rt.month, freq="M")
COL2SC = {1: 14, 2: 20, 3: 27, 4: 35, 5: 40}   # 只有燈號、沒分數時用該燈號區間中點
score_rt = rt.score_rt.fillna(rt.color_rt.map(COL2SC))
score_rt.loc["2013-06"] = 23 if pd.isna(rt.score_rt.get(pd.Period("2013-06", "M"))) or True else None  # 解析錯誤：當月燈號轉綠，分數≥23，標題寫「轉亮綠燈」
lead_rt = rt.lead_dir_rt

def light(s):
    return pd.cut(s, [0, 16.5, 22.5, 31.5, 37.5, 46], labels=[1, 2, 3, 4, 5]).astype(float)

def make(score, lead, px, pub_lag):
    """score/lead 以資料月為索引；pub_lag=1：資料月 t 在 t+1 月底可用"""
    sig = pd.DataFrame({"score": score, "lead": lead})
    sig.index = sig.index + pub_lag
    df = pd.DataFrame({"px": px}).join(sig)
    df["ret_next"] = df.px.pct_change().shift(-1)
    df["fwd12"] = np.log(df.px.shift(-12) / df.px); df["fwd6"] = np.log(df.px.shift(-6) / df.px)
    df["light"] = light(df.score)
    return df

def a2_rule(lt):
    pos, out = 1.0, []
    for v in lt:
        if v == 1: pos = 1.0
        elif v == 5: pos = 0.0
        out.append(pos)
    return pd.Series(out, index=lt.index)

def run(df, title, phase_med=None):
    P(f"\n=========== {title} ===========")
    P("樣本:", df.score.first_valid_index(), "~", df.index[-1], " 有訊號月數:", df.score.notna().sum())
    res = {}
    z = df[["score", "fwd12"]].dropna()
    b, t = nw_t(z.fwd12.values, z.score.values); r, p1 = circ_p(z.fwd12.values, z.score.values); p2 = 2 * min(p1, 1 - p1)
    P(f"A1 分數→未來12月：每分 {b*100:.2f}%/年, NW t={t:.2f} (雙尾p={2*(1-stats.norm.cdf(abs(t))):.4f}), 循環平移雙尾p={p2:.4f}, N={len(z)}")
    g = df.dropna(subset=["light", "fwd12"]).groupby("light")
    tab = pd.DataFrame({"月數": g.size(), "未來12月平均": g.fwd12.mean().map(lambda v: f"{np.expm1(v):.1%}"),
                        "未來12月跌>15%機率": g.fwd12.apply(lambda v: f"{(v < np.log(.85)).mean():.0%}")})
    tab.index = tab.index.map({1: "藍", 2: "黃藍", 3: "綠", 4: "黃紅", 5: "紅"}); P(tab.to_string())
    # A2
    lt = df.light.ffill(); e = a2_rule(lt)
    rr = (e * (df.ret_next + DIV) + (1 - e) * CASH).dropna(); bh = (df.ret_next + DIV).dropna()
    rr, bh = rr.loc[lt.first_valid_index():], bh.loc[lt.first_valid_index():]
    s1, s0 = stats_of(rr), stats_of(bh)
    P(f"A2 藍買紅賣：年化={s1['CAGR']:.2%} 最大回撤={s1['MDD']:.1%} 平均曝險={e.loc[rr.index].mean():.2f} ｜ 買進持有：年化={s0['CAGR']:.2%} 最大回撤={s0['MDD']:.1%}")
    # A3
    for h in ("fwd6", "fwd12"):
        z = df[["lead", h]].dropna(); z = z[z.lead != 0]
        b, t = nw_t(z[h].values, z.lead.values, 12 if h == "fwd12" else 6); r, p1 = circ_p(z[h].values, z.lead.values)
        up, dn = z[z.lead > 0][h].mean(), z[z.lead < 0][h].mean()
        P(f"A3 領先指標方向→{h}: 上升時 {np.expm1(up):.1%} / 下降時 {np.expm1(dn):.1%}, NW t={t:.2f}, 循環平移雙尾p={2*min(p1,1-p1):.4f}, N={len(z)}")
    # A4 趨緩
    s = df.score
    med = s.expanding(36).median() if phase_med is None else phase_med.reindex(s.index)
    slow = (s >= med) & (s < s.shift(3))
    z = pd.DataFrame({"slow": slow, "f": df.fwd12}).dropna(); z = z[s.loc[z.index].notna() & s.shift(3).loc[z.index].notna()]
    m = z.slow.values.astype(bool); f = z.f.values; obs = f[~m].mean() - f[m].mean(); sims = []
    for _ in range(5000):
        k = rng.integers(12, len(m) - 12); mm = np.roll(m, k); sims.append(f[~mm].mean() - f[mm].mean())
    P(f"A4 趨緩(分數≥中位數且低於3月前) 月數={m.sum()}：未來12月 {np.expm1(f[m].mean()):.1%} vs 其他 {np.expm1(f[~m].mean()):.1%}, 循環平移單尾p={(np.sum(np.array(sims)>=obs)+1)/5001:.4f}")

# 1) 當時公布值（新聞），缺漏月沿用上次公布值
df_rt = make(score_rt.reindex(pd.period_range("2009-10", "2026-09", freq="M")).ffill(),
             lead_rt.reindex(pd.period_range("2009-10", "2026-09", freq="M")).ffill(), mepx, 1)
med_full = score_rev.expanding(36).median(); med_full.index = med_full.index + 1   # 中位數用 1984 年以來官方歷史（較長，避免短樣本偏差）
run(df_rt, "B1 當時公布值（新聞）2009-10 起，缺漏沿用上次", med_full)
# 2) 同期間改用修正後資料（衡量後見之明差異）
df_rv = make(score_rev.loc["2009-10":], lead_dir_rev.loc["2009-10":], mepx, 1)
run(df_rv, "B2 同期間：今天下載的修正後資料", med_full)
# 3) 長期：修正後資料 1984~（含後見之明，僅供參考）。1998 前用月平均指數 → 多延後 1 個月
df_long = make(score_rev.loc["1984-01":], lead_dir_rev.loc["1984-01":], avgpx, 2).loc["1985-01":]
run(df_long, "B3 長期 1984~：修正後資料（含後見之明，僅供參考；月平均指數多延後1月）")
df_long2 = make(score_rev, lead_dir_rev, mepx, 1).loc["1998-01":]
run(df_long2, "B4 1998~：修正後資料、月底收盤")
# 當時 vs 修正 差異
d = pd.DataFrame({"rt": rt.score_rt, "rev": score_rev}).dropna(); d.loc["2013-06", "rt"] = np.nan; d = d.dropna()
P(f"\n修正幅度（{len(d)} 個月有當時分數）：完全相同 {(d.rt==d.rev).mean():.0%}，平均絕對差 {(d.rt-d.rev).abs().mean():.2f} 分，燈號相同 {(light(d.rt)==light(d.rev)).mean():.0%}")
dl = pd.DataFrame({"rt": rt.lead_dir_rt, "rev": lead_dir_rev}).dropna(); dl = dl[dl.rt != 0]
P(f"領先指標方向（{len(dl)} 個月）當時與修正後相同比例：{(dl.rt == dl.rev).mean():.0%}")
(OUT / "official_test.txt").write_text("\n".join(lines), encoding="utf-8")
