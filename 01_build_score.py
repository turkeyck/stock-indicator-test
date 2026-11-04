"""第四輪：多指標景氣分數（依 00_預先登記.md 實作）"""
import json, sys
import numpy as np, pandas as pd
from pathlib import Path

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)

def ndc(f):
    d = pd.read_csv(RAW / "ndc" / f, na_values=["-", ""])
    d.index = pd.PeriodIndex(d.pop("Date").astype(str), freq="M")
    return d.apply(pd.to_numeric, errors="coerce")

lead, sig, lag_, coin, comp = (ndc("領先指標構成項目.csv"), ndc("景氣對策信號構成項目.csv"),
                               ndc("落後指標構成項目.csv"), ndc("同時指標構成項目.csv"), ndc("景氣指標與燈號.csv"))

def fred(s, how="mean"):
    d = pd.read_csv(RAW / "fred" / f"{s}.csv", na_values=["."])
    d.columns = ["date", "v"]; d["date"] = pd.to_datetime(d["date"])
    x = d.set_index("date")["v"].astype(float).dropna()
    x = x.resample("ME").last() if how == "last" else x.resample("ME").mean()
    x.index = x.index.to_period("M"); return x

def yoy(x, n=12): return x / x.shift(n) - 1
def sum3(x): return x.rolling(3).sum()
def avg3(x): return x.rolling(3).mean()

# ---------- 原始轉換（每個 entry: (series, kind, sign, lag)）----------
# kind: 'growth' -> 年增率水準 + 年增率6月變化 ; 'levelchg' -> 水準 + 6月變化 ; 'level' -> 只用水準
ind = {}
ind["T1_外銷訂單動向"] = (avg3(lead.iloc[:, 0]), "levelchg", +1, 1)
ind["T2_海關出口"] = (yoy(sum3(sig["海關出口值(十億元)"])), "growth", +1, 1)
ind["T3_M1B"] = (yoy(sig["貨幣總計數M1B(百萬元)"]), "growth", +1, 1)
ind["T4_工業生產"] = (yoy(avg3(sig["工業生產指數(Index2021=100)"])), "growth", +1, 1)
ind["T5_半導體設備進口"] = (yoy(sum3(lead.iloc[:, 5])), "growth", +1, 1)
ind["T6_機電設備進口"] = (yoy(sum3(sig["機械及電機設備進口值(十億元)"])), "growth", +1, 1)
ind["T7_加班工時"] = (yoy(avg3(sig["工業及服務業加班工時(小時)"])), "growth", +1, 2)
ind["T8_員工淨進入率"] = (avg3(lead.iloc[:, 3]), "levelchg", +1, 2)
u = avg3(lag_["失業率(%)"]); ind["T9_台灣失業Sahm"] = (u - lag_["失業率(%)"].rolling(12).min(), "level", -1, 1)
r = lag_["五大銀行新承做放款平均利率(年息百分比)"]; ind["T10_放款利率12月變化"] = (r - r.shift(12), "level", -1, 1)
fx = fred("DEXTAUS", "last"); ind["T11_台幣貶值3月"] = (fx / fx.shift(3) - 1, "level", -1, 0)
ind["G1_南韓出口"] = (yoy(sum3(fred("XTEXVA01KRM667S"))), "growth", +1, 1)
ind["G2_美殖利率曲線"] = (fred("T10Y3M", "last"), "level", +1, 0)
b = fred("BAA10Y", "last"); ind["G3_信用利差6月變化"] = (b - b.shift(6), "level", -1, 0)
U = fred("UNRATE"); ind["G4_美失業Sahm"] = (avg3(U) - U.rolling(12).min(), "level", -1, 1)
ic = pd.read_csv(RAW / "fred" / "ICSA.csv"); ic.columns = ["d", "v"]; ic["d"] = pd.to_datetime(ic["d"])
ic = ic.set_index("d")["v"].astype(float).rolling(4).mean().resample("ME").last(); ic.index = ic.index.to_period("M")
ind["G5_初領失業金年增"] = (yoy(ic), "level", -1, 0)
ind["G6_核心資本財訂單"] = (yoy(sum3(fred("NEWORDER"))), "growth", +1, 1)
ind["G7_美建照"] = (yoy(avg3(fred("PERMIT"))), "growth", +1, 1)
ind["G8_銅價年增"] = (yoy(fred("PCOPPUSDM")), "level", +1, 1)
ff = fred("FEDFUNDS"); ind["G9_聯邦利率12月變化"] = (ff - ff.shift(12), "level", -1, 1)
oil = fred("DCOILWTICO", "last"); ind["G10_油價年增"] = (yoy(oil), "level", -1, 0)

def exp_pct(x, minp=60):
    """擴張視窗百分位：每個月只跟到當月為止的歷史比較，回傳 [-1,1]"""
    x = x.dropna(); out = pd.Series(np.nan, index=x.index); v = x.values
    for i in range(minp - 1, len(v)):
        h = v[: i + 1]; out.iloc[i] = ((h < v[i]).sum() + 0.5 * ((h == v[i]).sum() - 1)) / (len(h) - 1)
    return out * 2 - 1

scores = {}
for k, (s, kind, sign, lag) in ind.items():
    s = s.replace([np.inf, -np.inf], np.nan).dropna() * sign
    if kind == "level": sc = exp_pct(s)
    else:
        sc = pd.concat([exp_pct(s), exp_pct((s - s.shift(6)).dropna())], axis=1).mean(axis=1, skipna=False)
    sc.index = sc.index + lag  # 公布落後：資料月 t 在 t+lag 月底才可用
    scores[k] = sc
S = pd.DataFrame(scores).sort_index()
S = S.loc["1984-01":"2026-10"]
n_av = S.notna().sum(axis=1)
comp_score = S.mean(axis=1).where(n_av >= 5)

def exp_level(x, minp=36):
    p = exp_pct(x.dropna(), minp)
    return pd.cut(p, [-1.01, -0.6, -0.2, 0.2, 0.6, 1.01], labels=[1, 2, 3, 4, 5]).astype(float)

level = exp_level(comp_score)
res = pd.DataFrame({"composite": comp_score, "n_ind": n_av, "level": level}).join(S)
res.index = res.index.astype(str)
res.to_csv(OUT / "score_monthly.csv", encoding="utf-8-sig")
print(res[["composite", "n_ind", "level"]].dropna().tail(15))
print("各指標起始月：\n", S.apply(lambda c: c.first_valid_index()))
