"""美股特徵（約 60 個）：擴張視窗百分位；修正型總經用 ALFRED 當時版本"""
import json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; RAW = HERE / "raw"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

def fred(s, how="mean"):
    d = pd.read_csv(RAW / "fred" / f"{s}.csv", na_values=["."]); d.columns = ["date", "v"]; d["date"] = pd.to_datetime(d["date"])
    x = d.set_index("date")["v"].astype(float).dropna()
    x = x.resample("ME").last() if how == "last" else x.resample("ME").mean(); x.index = x.index.to_period("M"); return x.dropna()
def lagged(x, L): y = x.copy(); y.index = y.index + L; return y
yoy = lambda x, n=12: x / x.shift(n) - 1

# ---- 價格 ----
j = json.load(open(RAW / "gspc_d.json"))["chart"]["result"][0]
d = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s")).dropna()
px = d.resample("ME").last(); px.index = px.index.to_period("M")
dr = np.log(d).diff(); rv = dr.groupby(dr.index.to_period("M")).std() * np.sqrt(252)
sh = pd.read_excel(RAW / "ie_data.xls", sheet_name="Data", header=7)
sh = sh[pd.to_numeric(sh["Date"], errors="coerce").notna()].copy()
sh["Date"] = sh["Date"].astype(float).map(lambda v: pd.Period(f"{int(v)}-{int(round((v - int(v)) * 100)):02d}", "M"))
sh = sh.set_index("Date")
D = pd.to_numeric(sh["D"], errors="coerce"); E = pd.to_numeric(sh["E"], errors="coerce"); CAPE = pd.to_numeric(sh["CAPE"], errors="coerce")
tb = fred("TB3MS"); g10 = fred("GS10"); ff = fred("FEDFUNDS"); aaa = fred("AAA"); baa = fred("BAA"); cpi = fred("CPIAUCNS")
F = {}
def add(name, x, sign, lag=0): F[name] = (lagged(x.replace([np.inf, -np.inf], np.nan).dropna(), lag), sign)
# 估值（盈餘落後 6 個月；股息落後 3 個月）
add("V1_CAPE", CAPE, -1, 6)
add("V2_股息殖利率", (D / px).dropna(), +1, 3)
add("V3_盈餘殖利率", (lagged(E, 6) / px).dropna(), +1, 0)
add("V4_盈餘殖利率減10年債", (lagged(E, 6) / px * 100 - g10).dropna(), +1, 0)
add("V5_盈餘年增", yoy(E), +1, 6)
# 利率
add("R1_10年債水準", g10, -1, 0); add("R2_10年債12月變化", g10 - g10.shift(12), -1, 0)
add("R3_國庫券12月變化", tb - tb.shift(12), -1, 0); add("R4_聯邦利率12月變化", ff - ff.shift(12), -1, 0)
add("R5_聯邦利率6月變化", ff - ff.shift(6), -1, 0)
ts = g10 - tb; add("R6_期限利差10y-3m", ts, +1, 0); add("R7_期限利差12月變化", ts - ts.shift(12), +1, 0)
infl = yoy(cpi) * 100; add("R8_實質利率", g10 - lagged(infl, 1), -1, 0)
g2 = fred("GS2"); add("R9_10y-2y", g10 - g2, +1, 0)
# 信用
q = baa - aaa; add("C1_BAA-AAA", q, -1, 0); add("C2_BAA-AAA6月變化", q - q.shift(6), -1, 0)
c3 = baa - g10; add("C3_BAA-10年債", c3, -1, 0); add("C4_BAA-10年債6月變化", c3 - c3.shift(6), -1, 0)
# 通膨與商品
add("I1_CPI年增", infl, -1, 1); add("I2_CPI年增12月變化", infl - infl.shift(12), -1, 1)
add("I3_PPI年增", yoy(fred("PPIACO")), -1, 1)
oil = fred("WTISPLC"); add("I4_油價年增", yoy(oil), -1, 1); add("I5_油價6月變化", oil / oil.shift(6) - 1, -1, 1)
# 房市（建照：現行資料，落後 1）
add("H2_建照年增", yoy(fred("PERMIT").rolling(3).mean()), +1, 1)
ic = pd.read_csv(RAW / "fred" / "ICSA.csv"); ic.columns = ["d", "v"]; ic["d"] = pd.to_datetime(ic["d"])
ic = ic.set_index("d")["v"].astype(float).rolling(4).mean().resample("ME").last(); ic.index = ic.index.to_period("M")
add("L6_初領失業金年增", yoy(ic), -1, 0); add("L7_初領對52週低點", ic / ic.rolling(12).min() - 1, -1, 0)
um = fred("UMCSENT"); add("S1_信心12月變化", um - um.shift(12), +1, 1); add("S2_信心水準", um, +1, 1)

# ---- ALFRED 當時版本 ----
RTDEF = {  # 名稱: (series, function(vintage series)->value, sign)
    "L1_Sahm失業": ("UNRATE", lambda x: x.iloc[-3:].mean() - x.iloc[-14:-2].min(), -1),
    "L2_失業率12月變化": ("UNRATE", lambda x: x.iloc[-1] - x.iloc[-13], -1),
    "L3_非農年增": ("PAYEMS", lambda x: x.iloc[-1] / x.iloc[-13] - 1, +1),
    "L4_非農3月年化": ("PAYEMS", lambda x: (x.iloc[-1] / x.iloc[-4]) ** 4 - 1, +1),
    "L5_製造業工時12月變化": ("AWHMAN", lambda x: x.iloc[-3:].mean() - x.iloc[-15:-12].mean(), +1),
    "P1_工業生產年增": ("INDPRO", lambda x: x.iloc[-1] / x.iloc[-13] - 1, +1),
    "P2_工業生產年增6月變化": ("INDPRO", lambda x: (x.iloc[-1] / x.iloc[-13]) - (x.iloc[-7] / x.iloc[-19]), +1),
    "H1_新屋開工年增": ("HOUST", lambda x: x.iloc[-3:].mean() / x.iloc[-15:-12].mean() - 1, +1),
    "H3_新屋開工6月變化": ("HOUST", lambda x: x.iloc[-3:].mean() / x.iloc[-9:-6].mean() - 1, +1),
    "M1_M2年增": ("M2SL", lambda x: x.iloc[-1] / x.iloc[-13] - 1, +1),
}
rtinfo = {}
for name, (s, fn, sign) in RTDEF.items():
    V = pd.read_csv(RAW / "alfred" / f"{s}.csv", index_col=0, na_values=["."]) if (RAW / "alfred" / f"{s}.csv").exists() else pd.DataFrame()
    vals = {}
    for c in V.columns:
        x = V[c].dropna()
        if len(x) < 20: continue
        vd = pd.Period(c.split("_")[-1][:6], "M")
        vals[vd] = fn(x)
    rt = pd.Series(vals).sort_index()
    # 沒有當時版本的早期月份：用現行資料，資料月 t 在 t+3 才用（保守）
    cur = fred(s); cv = {}
    for i in range(20, len(cur)): cv[cur.index[i] + 3] = fn(cur.iloc[: i + 1])
    cv = pd.Series(cv)
    first = rt.index.min() if len(rt) else None
    x = pd.concat([cv[cv.index < first] if first is not None else cv, rt]) if len(rt) else cv
    x = x[~x.index.duplicated(keep="last")].sort_index()
    F[name] = (x, sign); rtinfo[name] = str(first)
add("M2_實質M2年增", (F["M1_M2年增"][0] * 100 - lagged(infl, 1)).dropna(), +1, 0)
# 價格趨勢 / 波動
for n in (3, 6, 10, 12, 24): add(f"T{n}_對{n}月均", px / px.rolling(n).mean() - 1, +1, 0)
for n in (1, 3, 6, 12, 24): add(f"Q{n}_{n}月動能", px / px.shift(n) - 1, +1, 0)
add("Q12x_12月超額動能", (px / px.shift(12) - 1) * 100 - tb, +1, 0)
mr = px.pct_change(); add("W1_12月波動", mr.rolling(12).std(), -1, 0)
add("W2_當月實現波動", rv, -1, 0); add("W3_實現波動3月變化", rv - rv.shift(3), -1, 0)
add("W4_距12月高點", px / px.rolling(12).max() - 1, +1, 0); add("W5_距歷史高點", px / px.cummax() - 1, +1, 0)
add("W6_VIX", fred("VIXCLS", "last"), -1, 0)

def exp_pct(x, minp=60):
    v = x.values; out = np.full(len(v), np.nan)
    for i in range(minp - 1, len(v)):
        h = v[: i + 1]; out[i] = ((h < v[i]).sum() + 0.5 * ((h == v[i]).sum() - 1)) / (len(h) - 1)
    return pd.Series(out, index=x.index)
X = pd.DataFrame({k: exp_pct(s.sort_index()) for k, (s, _) in F.items()}).sort_index().loc["1960-01":"2026-10"]
X.index = X.index.astype(str); X.to_csv(OUT / "features.csv", encoding="utf-8-sig")
pd.Series({k: s for k, (_, s) in F.items()}).to_csv(OUT / "theory_sign.csv", encoding="utf-8-sig")
# 報酬：月底價 + 股息（Shiller D/12，2023-10 後 1.4%/年）；現金 TB3MS
dy = (D / px / 12).reindex(px.index).fillna(0.014 / 12)
R = pd.DataFrame({"r": px.shift(-1) / px - 1 + dy, "cash": tb.reindex(px.index).ffill() / 1200})
R.index = R.index.astype(str); R.to_csv(OUT / "returns.csv", encoding="utf-8-sig")
print(len(F), "個特徵；ALFRED 當時版本起始：", rtinfo)
print(X.apply(lambda c: c.first_valid_index()).sort_values().tail(15).to_string())
