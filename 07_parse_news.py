"""從新聞摘要解析國發會「當時公布」的景氣對策信號分數、燈號、領先指標方向"""
import json, re, glob, datetime
import pandas as pd
from pathlib import Path
HERE = Path(__file__).parent
NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}

def month_of(text, pub):
    """資料月份：文中「X月景氣」；年份由公布日推得（公布月的前 1~2 個月）"""
    m = re.search(r"(?:(\d{2,4})\s*年\s*)?(\d{1,2}|十一|十二|[一二三四五六七八九十])\s*月(?:份)?(?:的)?\s*(?:景氣|對策|燈號|整體景氣)", text)
    if not m: return None
    mo = int(m.group(2)) if m.group(2).isdigit() else NUM[m.group(2)]
    if not 1 <= mo <= 12: return None
    for back in (1, 2, 3):
        p = pd.Period(pub, "M") - back
        if p.month == mo: return p
    return None

COLOR = {"黃藍燈": 2, "黃紅燈": 4, "藍燈": 1, "綠燈": 3, "紅燈": 5}
def color_of(text):
    m = re.search(r"(?:燈號|信號)[^。，]{0,12}?(黃藍燈|黃紅燈|藍燈|綠燈|紅燈)", text) or re.search(r"(黃藍燈|黃紅燈|藍燈|綠燈|紅燈)", text)
    return COLOR[m.group(1)] if m else None

def score_of(text):
    m = re.search(r"(?:綜合判斷分數|對策信號分數|對策分數|綜合分數|判斷分數)[^。\d]{0,8}?(\d{1,2})\s*分", text)
    if m and 9 <= int(m.group(1)) <= 45: return int(m.group(1))
    return None

def lead_of(text):
    m = re.search(r"領先指標[^。]{0,40}?(上升|回升|續升|走升|揚升|增加|下滑|下跌|下降|續降|減少|續跌|轉跌|持平)", text)
    if not m: return None
    w = m.group(1); return 0 if w == "持平" else (1 if w in ("上升", "回升", "續升", "走升", "揚升", "增加") else -1)

rows = []
for f in sorted(glob.glob(str(HERE / "raw" / "news" / "d*.json"))):
    for x in json.load(open(f, encoding="utf-8")):
        bf = HERE / "raw" / "news" / "body" / f"{x['id']}.txt"
        body = bf.read_text(encoding="utf-8")[:2500] if bf.exists() else ""
        text = x["title"] + "。" + (x.get("summary") or "") + "。" + body
        if any(k in x["title"] for k in ("預告", "今將公布", "搶先讀", "今日必看")): continue
        pub = datetime.datetime.fromtimestamp(x["t"], datetime.timezone(datetime.timedelta(hours=8)))
        if not ("國發會" in text or "經建會" in text or "對策信號" in text or "景氣燈號" in text): continue
        if any(k in text for k in ("製造業景氣", "台經院", "電力景氣", "房地產", "PMI", "預估", "可望", "恐")) and score_of(text) is None: continue
        rows.append(dict(pub=pub.strftime("%Y-%m-%d %H:%M"), month=month_of(text, pub.strftime("%Y-%m")), score=score_of(text),
                         color=color_of(text), lead=lead_of(text), id=x["id"], title=x["title"], text=text[:600].replace("\n"," ")))
raw = pd.DataFrame(rows).sort_values("pub")
raw.to_csv(HERE / "out" / "news_parsed_all.csv", index=False, encoding="utf-8-sig")

# 每個資料月取「最早」有分數的報導（公布當下）；顏色、領先指標方向取該月最早出現者
ok = raw.dropna(subset=["month"])
first = []
for mo, g in ok.groupby("month"):
    g = g.sort_values("pub")
    gs = g.dropna(subset=["score"])
    # 只接受在資料月的下個月 20 日之後~再下個月 5 日之前 的報導（國發會發布窗口）
    def in_window(p): d = pd.Timestamp(p[:10]); lo = (mo + 1).to_timestamp() + pd.Timedelta(days=19); return lo <= d <= (mo + 2).to_timestamp() + pd.Timedelta(days=5)
    gs = gs[gs.pub.map(in_window)]; gw = g[g.pub.map(in_window)]
    if gw.empty: continue
    sc = gs.score.iloc[0] if len(gs) else None
    # 分數眾數檢查（多篇報導互相印證）
    mode_sc = gs.score.mode().iloc[0] if len(gs) else None
    col = gw.color.dropna().iloc[0] if gw.color.notna().any() else None
    ld = gw.lead.dropna().iloc[0] if gw.lead.notna().any() else None
    first.append(dict(month=str(mo), pub=gw.pub.iloc[0], score_rt=sc, score_mode=mode_sc, n_scored=len(gs), color_rt=col, lead_dir_rt=ld,
                      title=(gs.title.iloc[0] if len(gs) else gw.title.iloc[0])))
rt = pd.DataFrame(first)
rt.to_csv(HERE / "out" / "realtime_signals.csv", index=False, encoding="utf-8-sig")
print(rt.to_string(max_colwidth=40))
print("月數:", len(rt), "有分數:", rt.score_rt.notna().sum(), "有顏色:", rt.color_rt.notna().sum(), "有領先方向:", rt.lead_dir_rt.notna().sum(),
      "分數首篇≠眾數:", (rt.score_rt != rt.score_mode).sum())
