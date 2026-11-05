"""解析 French CSV，建立約 475 個只做多價值加權積木（已扣交易成本），存 out/blocks.csv"""
import re
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; RAW = HERE / "raw"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

def section(fname, title_pat="Value Weight(ed)? Returns"):
    fname = next(p.name for p in RAW.iterdir() if p.name.lower() == fname.lower())
    L = (RAW / fname).read_text(encoding="latin-1").splitlines()
    i = next(k for k, l in enumerate(L) if re.search(title_pat, l, re.I) and "Monthly" in l) if title_pat else next(k for k, l in enumerate(L) if l.strip().startswith(",Mkt-RF"))- 1
    hdr = [h.strip() for h in L[i + 1].split(",")]; rows = []
    for l in L[i + 2:]:
        p = [x.strip() for x in l.split(",")]
        if not p[0].isdigit() or len(p[0]) != 6: break
        rows.append(p)
    df = pd.DataFrame(rows, columns=hdr).set_index(hdr[0]).astype(float)
    df.index = pd.PeriodIndex(df.index, freq="M"); df[df <= -99] = np.nan
    return df / 100

ff = section("F-F_Research_Data_Factors.CSV", None)
mkt, rf = ff["Mkt-RF"] + ff["RF"], ff["RF"]
COST = {"Prior_1_0": .08, "Prior_12_2": .03, "VAR": .015, "RESVAR": .015, "BETA": .015, "Prior_60_13": .01}
DEC = ["Lo 10", "2-Dec", "3-Dec", "4-Dec", "5-Dec", "6-Dec", "7-Dec", "8-Dec", "9-Dec", "Hi 10"]
blocks, cost, group = {}, {}, {}
uni = {"ME": "Portfolios_Formed_on_ME.csv", "BEME": "Portfolios_Formed_on_BE-ME.csv", "OP": "Portfolios_Formed_on_OP.csv",
       "INV": "Portfolios_Formed_on_INV.csv", "EP": "Portfolios_Formed_on_E-P.csv", "CFP": "Portfolios_Formed_on_CF-P.csv",
       "DP": "Portfolios_Formed_on_D-P.csv", "AC": "Portfolios_Formed_on_AC.csv", "BETA": "Portfolios_Formed_on_BETA.csv",
       "NI": "Portfolios_Formed_on_NI.csv", "VAR": "Portfolios_Formed_on_VAR.csv", "RESVAR": "Portfolios_Formed_on_RESVAR.csv",
       "Prior_12_2": "10_Portfolios_Prior_12_2.csv", "Prior_1_0": "10_Portfolios_Prior_1_0.csv", "Prior_60_13": "10_Portfolios_Prior_60_13.csv"}
for k, f in uni.items():
    fn = next(p.name for p in RAW.iterdir() if p.name.lower() == f.lower())
    df = section(fn)
    cols = [c for c in DEC if c in df.columns]
    if len(cols) < 10:
        alt = ["Lo 10"] + [f"Dec {i}" for i in range(2, 10)] + ["Hi 10"]
        cols = alt if all(c in df.columns for c in alt) else list(df.columns[-10:])
    for j, c in enumerate(cols):
        n = f"{k}_D{j+1}"; blocks[n] = df[c]; group[n] = k
        cost[n] = COST.get(k, .005) + (.015 if k == "ME" and j < 2 else 0)
biv = {"ME×BEME": "25_Portfolios_5x5.csv", "BEME×INV": "25_Portfolios_BEME_INV_5x5.csv", "BEME×OP": "25_Portfolios_BEME_OP_5x5.csv",
       "ME×AC": "25_Portfolios_ME_AC_5x5.csv", "ME×BETA": "25_Portfolios_ME_BETA_5x5.csv", "ME×INV": "25_Portfolios_ME_INV_5x5.csv",
       "ME×NI": "25_Portfolios_ME_NI_5x5.csv", "ME×OP": "25_Portfolios_ME_OP_5x5.csv", "ME×Prior_12_2": "25_Portfolios_ME_Prior_12_2.csv",
       "ME×Prior_1_0": "25_Portfolios_ME_Prior_1_0.csv", "ME×Prior_60_13": "25_Portfolios_ME_Prior_60_13.csv",
       "ME×RESVAR": "25_Portfolios_ME_RESVAR_5x5.csv", "ME×VAR": "25_Portfolios_ME_VAR_5x5.csv", "OP×INV": "25_Portfolios_OP_INV_5x5.csv"}
for k, f in biv.items():
    fn = next(p.name for p in RAW.iterdir() if p.name.lower() == f.lower())
    df = section(fn, r"Value Weight|Average Value")
    sec = k.split("×")[1]
    for j, c in enumerate(df.columns[:25]):
        n = f"{k}_{j//5+1}{j%5+1}"; blocks[n] = df[c]; group[n] = k
        cost[n] = COST.get(sec, .005) + (.015 if k.startswith("ME×") and j // 5 == 0 else 0)
B = pd.DataFrame(blocks).loc["1963-07":"2026-09"]
C = pd.Series(cost)
Bnet = B - C / 12
Bnet["_MKT"] = mkt; Bnet["_RF"] = rf
Bnet.index = Bnet.index.astype(str); Bnet.to_csv(OUT / "blocks.csv", encoding="utf-8-sig")
pd.DataFrame({"group": pd.Series(group), "cost": C}).to_csv(OUT / "block_info.csv", encoding="utf-8-sig")
print("積木數:", B.shape[1], " 期間:", B.index[0], "~", B.index[-1])
print("缺值最多:", B.isna().sum().sort_values().tail(5).to_dict())
