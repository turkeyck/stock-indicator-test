"""建立約 60 個特徵（擴張視窗百分位，已套用公布落後），存 out/features.csv"""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; RAW = ROOT / "raw"; OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
# 重用第四輪的 21 個指標定義（ind: name -> (series, kind, sign, lag)）
src = (ROOT / "01_build_score.py").read_text(encoding="utf-8").split("def exp_pct")[0]
sys.argv = ["x", str(RAW)]; ns = {"__file__": str(ROOT / "01_build_score.py")}; exec(src, ns)
ind, ndc, fred, yoy, sum3, avg3 = ns["ind"], ns["ndc"], ns["fred"], ns["yoy"], ns["sum3"], ns["avg3"]
sig, lag_, coin, lead = ns["sig"], ns["lag_"], ns["coin"], ns["lead"]

F = {}   # name -> (series, theory_sign, lag)
for k, (s, kind, sign, lag) in ind.items():
    F[k] = (s, sign, lag)
    if kind in ("growth", "levelchg"): F[k + "_6m變化"] = (s - s.shift(6), sign, lag)
# 新增：台灣其他構成項目
F["T12_製造業銷售量年增"] = (yoy(avg3(coin["製造業銷售量指數(Index2021=100)"])), +1, 1)
F["T13_批發零售餐飲年增"] = (yoy(sum3(coin["批發、零售及餐飲業營業額(十億元)"])), +1, 1)
F["T14_用電量年增"] = (yoy(sum3(coin["電力(企業)總用電量(十億度)"])), +1, 1)
F["T15_建築開工年增"] = (yoy(sum3(lead.iloc[:, 4])), +1, 2)
F["T16_製造業存貨年增"] = (yoy(lag_["製造業存貨價值(千元)"]), -1, 2)
F["T17_單位勞動成本年增"] = (yoy(lag_["製造業單位產出勞動成本指數(2021=100)"]), -1, 2)
F["T18_放款與投資年增"] = (yoy(lag_["全體金融機構放款與投資(10億元)"]), +1, 1)
F["T19_存貨減出口成長"] = (yoy(lag_["製造業存貨價值(千元)"]) - yoy(sum3(sig["海關出口值(十億元)"])), -1, 2)
# 美國其他
F["G11_美10減2年"] = (fred("T10Y2Y", "last"), +1, 0)
F["G12_BAA利差水準"] = (fred("BAA10Y", "last"), -1, 0)
vix = fred("VIXCLS", "last"); F["G13_VIX水準"] = (vix, -1, 0); F["G14_VIX3月變化"] = (vix - vix.shift(3), -1, 0)
F["G15_密大信心年變化"] = (fred("UMCSENT") - fred("UMCSENT").shift(12), +1, 1)
F["G16_美CPI年增"] = (yoy(fred("CPIAUCSL")), -1, 1)
F["G17_美工業生產年增"] = (yoy(fred("INDPRO")), +1, 1)
F["G18_美製造業新訂單年增"] = (yoy(sum3(fred("AMTMNO"))), +1, 1)
F["G19_美耐久財訂單年增"] = (yoy(sum3(fred("DGORDER"))), +1, 1)
F["G20_初領失業金水準"] = (fred("ICSA", "last"), -1, 0)
fx = fred("DEXTAUS", "last"); F["G21_台幣12月貶值"] = (fx / fx.shift(12) - 1, -1, 0)
cu = fred("PCOPPUSDM"); F["G22_銅價6月變化"] = (cu / cu.shift(6) - 1, +1, 1)
oil = fred("DCOILWTICO", "last"); F["G23_油價6月變化"] = (oil / oil.shift(6) - 1, -1, 0)
nq = fred("NASDAQCOM", "last")
F["M1_那斯達克對10月均"] = (nq / nq.rolling(10).mean() - 1, +1, 0)
F["M2_那斯達克6月動能"] = (nq / nq.shift(6) - 1, +1, 0)
# 台股價格（月底收盤，1997-07 起；之前用月平均）
j = json.load(open(RAW / "price" / "twii.json"))["chart"]["result"][0]
d = pd.Series(j["indicators"]["quote"][0]["close"], index=pd.to_datetime(j["timestamp"], unit="s")).dropna()
px = d.resample("ME").last(); px.index = px.index.to_period("M")
avg = pd.Series(lead["股價指數(Index1966=100)"].values, index=lead.index).astype(float)
px = pd.concat([avg.loc[:"1997-06"] * (px.iloc[0] / avg.loc["1997-07"]), px])  # 早期接月平均，只用於特徵暖身
for n in (3, 6, 10, 12): F[f"P{n}_台股對{n}月均"] = (px / px.rolling(n).mean() - 1, +1, 0)
for n in (1, 3, 6, 12): F[f"Q{n}_台股{n}月動能"] = (px / px.shift(n) - 1, +1, 0)
r = px.pct_change(); F["V1_台股12月波動"] = (r.rolling(12).std(), -1, 0)
F["V2_距12月高點"] = (px / px.rolling(12).max() - 1, +1, 0)

def exp_pct(x, minp=60):
    x = x.replace([np.inf, -np.inf], np.nan).dropna(); v = x.values; out = np.full(len(v), np.nan)
    for i in range(minp - 1, len(v)):
        h = v[: i + 1]; out[i] = ((h < v[i]).sum() + 0.5 * ((h == v[i]).sum() - 1)) / (len(h) - 1)
    return pd.Series(out, index=x.index)

cols, signs = {}, {}
for k, (s, sign, lag) in F.items():
    p = exp_pct(s); p.index = p.index + lag; cols[k] = p; signs[k] = sign
X = pd.DataFrame(cols).sort_index().loc["1990-01":"2026-10"]
X.index = X.index.astype(str); X.to_csv(OUT / "features.csv", encoding="utf-8-sig")
pd.Series(signs).to_csv(OUT / "theory_sign.csv", encoding="utf-8-sig")
px.index = px.index.astype(str); px.to_csv(OUT / "px.csv", encoding="utf-8-sig")
print(len(F), "個特徵"); print(X.apply(lambda c: c.first_valid_index()).sort_values().to_string())
