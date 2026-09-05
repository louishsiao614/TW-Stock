"""台股量化研究儀表板：前一交易日掃描與個股三面向分析。"""
from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

TWSE = "https://openapi.twse.com.tw/v1/exchangeReport"
TPEX = "https://www.tpex.org.tw/openapi/v1"
TAIFEX = "https://openapi.taifex.com.tw/v1"
HEADERS = {"User-Agent": "TW-Quant-Research/1.0 (educational dashboard)"}

# Official industry codes are useful to machines but not to an investment workflow.
INDUSTRY_LABELS = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維", "05": "電機機械",
    "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業", "11": "橡膠工業",
    "12": "汽車工業", "14": "建材營造", "15": "航運業", "16": "觀光餐旅", "17": "金融保險",
    "18": "貿易百貨", "20": "其他", "21": "化學工業", "22": "生技醫療", "23": "油電燃氣",
    "24": "半導體業", "25": "電腦及週邊設備", "26": "光電業", "27": "通信網路", "28": "電子零組件",
    "29": "電子通路", "30": "資訊服務", "31": "其他電子業", "32": "文化創意", "33": "農業科技",
    "34": "數位雲端", "35": "綠能環保", "36": "運動休閒", "37": "居家生活", "80": "管理股票",
}
# High-signal sub-themes.  Other companies retain their official Chinese industry label.
THEME_OVERRIDES = {
    "2327": "被動元件", "2375": "被動元件", "2456": "被動元件", "2472": "被動元件",
    "2478": "被動元件", "2492": "被動元件", "3026": "被動元件", "3042": "被動元件",
    "3090": "被動元件", "3321": "被動元件", "3357": "被動元件", "6127": "被動元件",
    "6139": "被動元件", "6173": "被動元件", "8042": "被動元件",
}


def _number(value: object) -> float:
    """Convert API number strings (including commas and '--') safely."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).replace(",", "").replace("--", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def chinese_industry(value: object) -> str:
    """Translate official numeric industry codes while preserving supplied Chinese names."""
    text = str(value).strip()
    return INDUSTRY_LABELS.get(text.zfill(2), text if text and text != "nan" else "未分類")


def _get_json(url: str) -> list[dict]:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        return payload
    # TWSE's report endpoint returns column labels plus row arrays.
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        fields = payload.get("fields", [])
        return [dict(zip(fields, row)) for row in payload["data"]]
    return []


@st.cache_data(ttl=60 * 60, show_spinner=False)
def twse_daily() -> pd.DataFrame:
    rows = _get_json(f"{TWSE}/STOCK_DAY_ALL")
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    # OpenAPI field labels are Chinese; selecting defensively also preserves future fields.
    df = pd.DataFrame({
        "code": raw.get("Code", raw.get("證券代號")),
        "name": raw.get("Name", raw.get("證券名稱")),
        "industry": raw.get("Industry", raw.get("產業別", "其他")),
        "close": raw.get("ClosingPrice", raw.get("收盤價")),
        "change": raw.get("Change", raw.get("漲跌價差")),
        "volume": raw.get("TradeVolume", raw.get("成交股數")),
        "value": raw.get("TradeValue", raw.get("成交金額")),
        "open": raw.get("OpeningPrice", raw.get("開盤價")),
        "high": raw.get("HighestPrice", raw.get("最高價")),
        "low": raw.get("LowestPrice", raw.get("最低價")),
    })
    for col in ["close", "change", "volume", "value", "open", "high", "low"]:
        df[col] = df[col].map(_number)
    df["industry"] = df["industry"].fillna("其他")
    df["market"] = "上市"
    return df.dropna(subset=["code", "close"])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def tpex_daily() -> pd.DataFrame:
    # TPEx publishes the latest daily close through this versioned OpenAPI route.
    rows = _get_json(f"{TPEX}/tpex_mainboard_daily_close_quotes")
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    def field(*names: str) -> pd.Series:
        for name in names:
            if name in raw:
                return raw[name]
        return pd.Series(index=raw.index, dtype="object")
    df = pd.DataFrame({
        "code": field("SecuritiesCompanyCode", "證券代號"),
        "name": field("CompanyName", "證券名稱"),
        "industry": field("Industry", "產業別").fillna("上櫃其他"),
        "close": field("Close", "收盤"),
        "change": field("Change", "漲跌"),
        "volume": field("TradingShares", "成交股數"),
        "value": field("TransactionAmount", "成交金額"),
        "open": field("Open", "開盤"),
        "high": field("High", "最高"),
        "low": field("Low", "最低"),
    })
    for col in ["close", "change", "volume", "value", "open", "high", "low"]:
        df[col] = df[col].map(_number)
    df["market"] = "上櫃"
    return df.dropna(subset=["code", "close"])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def valuation() -> pd.DataFrame:
    rows = _get_json(f"{TWSE}/BWIBBU_ALL")
    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(columns=["code", "pe", "yield", "pb"])
    result = pd.DataFrame({
        "code": raw.get("Code", raw.get("證券代號")),
        "pe": raw.get("PEratio", raw.get("本益比")),
        "yield": raw.get("DividendYield", raw.get("殖利率(%)")),
        "pb": raw.get("PBratio", raw.get("股價淨值比")),
    })
    for col in ["pe", "yield", "pb"]:
        result[col] = result[col].map(_number)
    return result


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def company_profiles() -> pd.DataFrame:
    """Official company master data supplies the industry missing from daily quotes."""
    profiles = []
    for url, market_name in (
        ("https://openapi.twse.com.tw/v1/opendata/t187ap03_L", "上市"),
        (f"{TPEX}/mopsfin_t187ap03_O", "上櫃"),
    ):
        try:
            raw = pd.DataFrame(_get_json(url))
            if raw.empty:
                continue
            def profile_field(*names: str) -> pd.Series:
                for name in names:
                    if name in raw.columns:
                        return raw[name]
                return pd.Series(index=raw.index, dtype="object")
            profiles.append(pd.DataFrame({
                "code": profile_field("公司代號", "CompanyCode", "SecuritiesCompanyCode").astype(str),
                "industry": profile_field("產業別", "Industry", "IndustryName").map(chinese_industry),
                "market": market_name,
            }))
        except requests.RequestException:
            continue
    return pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame(columns=["code", "industry", "market"])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def institutional_daily(market_name: str) -> pd.DataFrame:
    """Latest published per-stock three-institution flow from the market operator."""
    url = "https://www.twse.com.tw/rwd/zh/fund/T86?date=&selectType=ALL&response=json" if market_name == "上市" else f"{TPEX}/tpex_3insti_daily_trading"
    raw = pd.DataFrame(_get_json(url))
    if raw.empty:
        return pd.DataFrame(columns=["code", "foreign_net", "trust_net", "dealer_net", "total_net"])
    def column(*names: str) -> pd.Series:
        for name in names:
            if name in raw.columns:
                return raw[name]
        return pd.Series(0.0, index=raw.index)
    foreign = column("ForeignTotalDifference", "ForeignNet", "外資及陸資買賣超股數", "外陸資買賣超股數(不含外資自營商)")
    trust = column("InvestmentTrustDifference", "InvestmentTrustNet", "投信買賣超股數")
    dealer = column("DealerDifference", "DealerTotalDifference", "DealerNet", "自營商買賣超股數")
    result = pd.DataFrame({
        "code": column("Code", "SecuritiesCompanyCode", "證券代號").astype(str),
        "foreign_net": foreign.map(_number), "trust_net": trust.map(_number), "dealer_net": dealer.map(_number),
    })
    official_total = column("TotalDifference", "TotalNet", "三大法人買賣超股數").map(_number)
    result["total_net"] = official_total.where(official_total.notna(), result[["foreign_net", "trust_net", "dealer_net"]].sum(axis=1))
    return result


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def derivative_flags() -> tuple[set[str], set[str], set[str]]:
    """Return current official underlying lists for stock futures, calls and puts."""
    futures, call_warrants, put_warrants = set(), set(), set()
    try:
        raw = pd.DataFrame(_get_json(f"{TAIFEX}/SSFLists"))
        for col in ("StockCode", "證券代號", "UnderlyingSecurityCode", "標的證券代號"):
            if col in raw:
                futures = set(raw[col].astype(str).str.extract(r"(\d{4,6})", expand=False).dropna())
                break
    except requests.RequestException:
        pass
    try:
        raw = pd.DataFrame(_get_json("https://openapi.twse.com.tw/v1/opendata/t187ap37_L"))
        if "標的證券/指數" in raw and "權證類型" in raw:
            underlying = raw["標的證券/指數"].astype(str).str.strip()
            types = raw["權證類型"].astype(str)
            # The official warrant master uses the *underlying name* (e.g. 台積電), not its code.
            call_warrants = set(underlying[types.str.contains("認購", na=False)])
            put_warrants = set(underlying[types.str.contains("認售", na=False)])
    except requests.RequestException:
        pass
    return futures, call_warrants, put_warrants


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def twse_institutional_history(code: str, days: int = 10) -> pd.DataFrame:
    """Latest ten available TWSE sessions for an individual security's three institutions."""
    rows = []
    for day in pd.bdate_range(end=date.today(), periods=days + 8, freq="B")[::-1]:
        if len(rows) >= days:
            break
        try:
            raw = pd.DataFrame(_get_json(f"https://www.twse.com.tw/rwd/zh/fund/T86?date={day:%Y%m%d}&selectType=ALL&response=json"))
            hit = raw[raw.get("證券代號", pd.Series(dtype=str)).astype(str) == str(code)]
            if hit.empty:
                continue
            item = hit.iloc[0]
            rows.append({"日期": day.date(), "外資": _number(item.get("外陸資買賣超股數(不含外資自營商)")),
                         "投信": _number(item.get("投信買賣超股數")), "自營商": _number(item.get("自營商買賣超股數")),
                         "三大法人": _number(item.get("三大法人買賣超股數"))})
        except requests.RequestException:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def twse_institutional_streak(days: int = 3) -> pd.DataFrame:
    """Find listed shares with official investment-trust net buys on consecutive trading days."""
    frames: list[pd.DataFrame] = []
    # Request extra calendar days so weekends/holidays do not break a three-session streak.
    for day in pd.bdate_range(end=date.today(), periods=days + 5, freq="B")[::-1]:
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?date={day:%Y%m%d}&selectType=ALL&response=json"
        try:
            raw = pd.DataFrame(_get_json(url))
            if raw.empty:
                continue
            trust_col = "投信買賣超股數"
            if trust_col not in raw:
                continue
            frame = pd.DataFrame({"code": raw["證券代號"].astype(str), "trust_net": raw[trust_col].map(_number)})
            frames.append(frame[frame.trust_net > 0])
            if len(frames) == days:
                break
        except requests.RequestException:
            continue
    if len(frames) < days:
        return pd.DataFrame(columns=["code", "trust_buy_days", "trust_net_sum"])
    combined = pd.concat(frames, ignore_index=True)
    return (combined.groupby("code").agg(trust_buy_days=("trust_net", "count"), trust_net_sum=("trust_net", "sum"))
            .reset_index().query("trust_buy_days == @days"))


def latest_market() -> pd.DataFrame:
    frames = []
    for fetch in (twse_daily, tpex_daily):
        try:
            frame = fetch()
            if not frame.empty:
                frames.append(frame)
        except requests.RequestException:
            continue
    if not frames:
        raise RuntimeError("暫時無法取得 TWSE/TPEx 資料，請稍後再試。")
    data = pd.concat(frames, ignore_index=True)
    profiles = company_profiles()
    if not profiles.empty:
        data = data.drop(columns=["industry"]).merge(profiles, how="left", on=["code", "market"])
        data["industry"] = data["industry"].fillna("未分類")
    data["theme"] = data["code"].astype(str).map(THEME_OVERRIDES).fillna(data["industry"])
    data["return_pct"] = data["change"] / (data["close"] - data["change"]) * 100
    data.loc[~np.isfinite(data["return_pct"]), "return_pct"] = np.nan
    return data


def theme_ranking(data: pd.DataFrame) -> pd.DataFrame:
    eligible = data[(data["value"] >= 30_000_000) & data["return_pct"].notna()].copy()
    group = eligible.groupby("theme", dropna=False).agg(
        stocks=("code", "count"), value=("value", "sum"), avg_return=("return_pct", "mean"),
        advancers=("return_pct", lambda x: (x > 0).mean()), volume=("volume", "sum")
    ).reset_index()
    # Percentile normalization makes dimensions comparable, while requiring breadth.
    group["value_score"] = group["value"].rank(pct=True)
    group["momentum_score"] = group["avg_return"].rank(pct=True)
    group["breadth_score"] = group["advancers"].rank(pct=True)
    group["theme_score"] = 100 * (0.45 * group.value_score + 0.35 * group.momentum_score + 0.20 * group.breadth_score)
    return group[group.stocks >= 2].sort_values("theme_score", ascending=False)


def resolve_symbol(query: str, universe: pd.DataFrame) -> tuple[str, str, str]:
    query = query.strip()
    hit = universe[(universe.code.astype(str) == query) | (universe.name.astype(str).str.contains(query, case=False, na=False))]
    if hit.empty:
        raise ValueError("找不到此股票代號或名稱，請輸入上市／上櫃普通股代號或名稱。")
    item = hit.sort_values("value", ascending=False).iloc[0]
    suffix = ".TW" if item.market == "上市" else ".TWO"
    return str(item.code), str(item.name), str(item.code) + suffix


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def history(symbol: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("缺少 yfinance；請先執行 pip install -r requirements.txt")
    hist = yf.Ticker(symbol).history(period="1y", auto_adjust=True)
    if hist.empty:
        raise RuntimeError("找不到可用的歷史行情。")
    hist.columns = [str(x).lower() for x in hist.columns]
    return hist


def indicators(hist: pd.DataFrame) -> pd.DataFrame:
    out = hist.copy()
    close = out.close
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["bb_upper"] = out["ma20"] + 2 * close.rolling(20).std()
    out["bb_lower"] = out["ma20"] - 2 * close.rolling(20).std()
    delta = close.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), -delta.clip(upper=0).rolling(14).mean()
    out["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["vol20"] = out.volume.rolling(20).mean()
    return out


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def batch_technical_signals(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Batch-fetch price history and classify monthly-trend setups without per-stock requests."""
    if yf is None or not symbols:
        return pd.DataFrame(columns=["symbol", "setup"])
    data = yf.download(list(symbols), period="5mo", auto_adjust=True, group_by="ticker", progress=False, threads=True)
    rows = []
    for symbol in symbols:
        try:
            prices = data[symbol].dropna(subset=["Close"]).copy()
            prices.columns = [str(c).lower() for c in prices.columns]
            enriched = indicators(prices)
            last = enriched.iloc[-1]
            if len(enriched) < 65 or pd.isna(last.ma20) or pd.isna(last.ma60):
                continue
            # Strong: price and both moving averages rise; Pullback: it held a prior advance,
            # then returns within ±3% of the monthly (20-session) average without breaking it.
            ma20_slope = (last.ma20 / enriched.ma20.iloc[-6] - 1) * 100
            strong = last.close > last.ma20 > last.ma60 and last.close >= enriched.close.tail(20).max() * 0.97 and ma20_slope > 0
            prior_high = enriched.close.tail(15).max()
            pullback = (last.ma20 * 0.97 <= last.close <= last.ma20 * 1.03
                        and prior_high >= last.ma20 * 1.05 and last.close >= last.ma60)
            if strong or pullback:
                rows.append({"symbol": symbol, "setup": "連續強勢（月線上）" if strong else "強勢後拉回月線附近",
                             "close": last.close, "ma20": last.ma20, "ma60": last.ma60,
                             "distance_to_ma20": (last.close / last.ma20 - 1) * 100, "ma20_slope_5d": ma20_slope})
        except (KeyError, IndexError):
            continue
    return pd.DataFrame(rows)


def strategy_candidates(market: pd.DataFrame, scan_limit: int) -> pd.DataFrame:
    """Transparent candidate list: official trust streak + price/volume main-force proxy + monthly setup."""
    streak = twse_institutional_streak(3)
    if streak.empty:
        return pd.DataFrame()
    # "主力" is not a standardized public field.  Define it as liquid, positive price-volume participation.
    liquid = market[(market.market == "上市") & (market.value >= 100_000_000) & (market.return_pct > 0)].copy()
    liquid["main_force_proxy"] = liquid.value.rank(pct=True) * (liquid.return_pct.rank(pct=True))
    pool = liquid.sort_values("main_force_proxy", ascending=False).head(scan_limit).copy()
    pool["symbol"] = pool.code.astype(str) + ".TW"
    technical = batch_technical_signals(tuple(pool.symbol))
    if technical.empty:
        return pd.DataFrame()
    result = pool.merge(streak, on="code").merge(technical.drop(columns=["close"], errors="ignore"), on="symbol")
    result["candidate_score"] = (result.main_force_proxy.rank(pct=True) * 35 +
                                 result.trust_net_sum.rank(pct=True) * 35 +
                                 (100 - result.distance_to_ma20.abs().clip(upper=10) * 10) * 0.20 +
                                 result.ma20_slope_5d.rank(pct=True) * 10)
    return result.sort_values("candidate_score", ascending=False)


def watchlist_recommendations(market: pd.DataFrame) -> pd.DataFrame:
    """Ten transparent daily research candidates, not personalised buy recommendations."""
    data = market[(market.value >= 100_000_000) & market.return_pct.notna()].copy()
    theme = theme_ranking(market)[["theme", "theme_score"]]
    data = data.merge(theme, on="theme", how="left")
    flows = []
    for market_name in ("上市", "上櫃"):
        try:
            f = institutional_daily(market_name).copy()
            f["market"] = market_name
            flows.append(f)
        except requests.RequestException:
            continue
    if flows:
        data = data.merge(pd.concat(flows)[["code", "market", "total_net", "trust_net"]], on=["code", "market"], how="left")
    else:
        data["total_net"], data["trust_net"] = 0, 0
    data[["total_net", "trust_net"]] = data[["total_net", "trust_net"]].fillna(0)
    data["quant_score"] = (data.value.rank(pct=True) * 35 + data.return_pct.rank(pct=True) * 30 +
                           data.theme_score.fillna(0).rank(pct=True) * 20 + data.total_net.rank(pct=True) * 15)
    def reason(row: pd.Series) -> str:
        notes = []
        if row.theme_score >= data.theme_score.quantile(.75): notes.append(f"{row.theme}題材熱度高")
        if row.return_pct >= data.return_pct.quantile(.75): notes.append("當日動能強")
        if row.total_net > 0: notes.append("三大法人買超")
        if row.trust_net > 0: notes.append("投信買超")
        return "、".join(notes) if notes else "成交額與相對強度位居市場前段"
    data["關注原因"] = data.apply(reason, axis=1)
    return data.sort_values("quant_score", ascending=False).head(10)


def technical_score(row: pd.Series) -> tuple[int, list[str]]:
    score, notes = 0, []
    if row.close > row.ma20 > row.ma60:
        score += 50; notes.append("收盤價位於 20/60 日均線之上，趨勢偏多")
    elif row.close < row.ma20 < row.ma60:
        score -= 50; notes.append("收盤價位於 20/60 日均線之下，趨勢偏弱")
    if 50 <= row.rsi14 <= 70:
        score += 20; notes.append("RSI 位於健康的偏多區")
    elif row.rsi14 > 75:
        score -= 15; notes.append("RSI 偏高，留意追價風險")
    if row.volume > row.vol20 * 1.2:
        score += 15; notes.append("成交量高於 20 日均量 20%")
    if pd.notna(row.bb_upper) and row.close >= row.bb_upper:
        score -= 10; notes.append("股價觸及／突破布林上軌，趨勢強但留意短線過熱")
    elif pd.notna(row.bb_lower) and row.close <= row.bb_lower:
        score -= 10; notes.append("股價觸及／跌破布林下軌，留意弱勢或反彈確認")
    else:
        notes.append("布林通道：股價位於正常波動區間")
    return int(np.clip(50 + score, 0, 100)), notes


def fundamental_score(v: pd.Series) -> tuple[int, list[str]]:
    score, notes = 50, []
    if pd.notna(v.pe) and 0 < v.pe <= 20:
        score += 20; notes.append(f"本益比 {v.pe:.1f} 倍")
    elif pd.notna(v.pe) and v.pe > 40:
        score -= 15; notes.append(f"本益比 {v.pe:.1f} 倍偏高")
    if pd.notna(v["yield"]) and v["yield"] >= 3:
        score += 15; notes.append(f"殖利率 {v['yield']:.2f}%")
    if pd.notna(v.pb) and 0 < v.pb <= 2:
        score += 15; notes.append(f"股價淨值比 {v.pb:.2f} 倍")
    return int(np.clip(score, 0, 100)), notes or ["公開估值欄位不足，請搭配最新財報判讀"]


def institutional_score(hist: pd.DataFrame, flow: pd.Series | None) -> tuple[int, list[str]]:
    """Prefer official flow; retain a labelled price-volume fallback if it is unavailable."""
    if flow is not None and pd.notna(flow.get("total_net")):
        foreign, trust, dealer, total = (flow.get(k, 0) for k in ("foreign_net", "trust_net", "dealer_net", "total_net"))
        score = 50 + (25 if total > 0 else -25) + (10 if trust > 0 else 0) + (5 if foreign > 0 else 0)
        notes = [f"三大法人合計買賣超：{total:,.0f} 股", f"外資：{foreign:,.0f}｜投信：{trust:,.0f}｜自營商：{dealer:,.0f} 股"]
        return int(np.clip(score, 0, 100)), notes
    recent = hist.tail(5)
    signed_volume = np.where(recent.close >= recent.open, recent.volume, -recent.volume).sum()
    volume_ratio = recent.volume.mean() / hist.volume.tail(25).mean()
    score = 50 + (20 if signed_volume > 0 else -20) + (10 if volume_ratio > 1 else 0)
    notes = ["法人明細暫不可得，改以近 5 日價量方向作籌碼代理指標",
             f"近 5 日量能／25 日均量：{volume_ratio:.2f} 倍"]
    return int(np.clip(score, 0, 100)), notes


def score_label(score: float) -> str:
    return "偏多研究訊號" if score >= 65 else "中性觀察" if score >= 45 else "偏弱／風險優先"


st.set_page_config(page_title="台股量化研究室", page_icon="📈", layout="wide")
st.title("台股量化研究室")
st.caption("以前一可得交易日收盤資料掃描市場；研究訊號不構成投資或股期交易建議。")

try:
    market = latest_market()
except Exception as exc:
    st.error(f"資料載入失敗：{exc}")
    st.stop()
futures_underlyings, call_warrant_underlyings, put_warrant_underlyings = derivative_flags()
market["個股期"] = market.code.astype(str).isin(futures_underlyings).map({True: "🟢 有個股期", False: "⚪ 無"})
market["認購權證"] = market.name.astype(str).isin(call_warrant_underlyings).map({True: "🟠 有認購", False: "⚪ 無"})
market["認售權證"] = market.name.astype(str).isin(put_warrant_underlyings).map({True: "🟣 有認售", False: "⚪ 無"})

requested_stock = st.query_params.get("stock", "")
requested_page = st.query_params.get("page", "每日主流族群")
pages = ["每日主流族群", "法人／月線選股", "個股三面向", "模型說明與風險"]
page = st.radio("功能頁面", pages, index=pages.index(requested_page) if requested_page in pages else 0, horizontal=True, label_visibility="collapsed")


def add_analysis_link(frame: pd.DataFrame, code_column: str) -> pd.DataFrame:
    result = frame.copy()
    result["個股分析"] = result[code_column].astype(str).map(lambda code: f"?stock={code}&page=個股三面向")
    return result


LINK_CONFIG = {"個股分析": st.column_config.LinkColumn("個股分析", display_text="🔎 開啟分析")}

if page == "每日主流族群":
    ranked = theme_ranking(market)
    st.subheader("主流族群熱度")
    st.caption("資料為 API 最近一次發布的完整收盤快照；非盤中資料。篩選：單檔成交額至少 3,000 萬元、族群至少 2 檔。")
    display = ranked.head(10).copy()
    display["成交額（億元）"] = display.value / 100_000_000
    display["上漲家數比"] = display.advancers * 100
    display = display.rename(columns={"theme": "族群", "stocks": "樣本數", "avg_return": "平均漲幅(%)", "theme_score": "熱度分數"})
    st.dataframe(display[["族群", "樣本數", "成交額（億元）", "平均漲幅(%)", "上漲家數比", "熱度分數"]].round(2), hide_index=True, use_container_width=True)
    if not ranked.empty:
        st.caption("點開下列族群即可檢視該族群個股與衍生商品標記。")
        for row in ranked.head(10).itertuples():
            with st.expander(f"{row.theme}｜{row.stocks} 檔個股"):
                leaders = market[market.theme == row.theme].sort_values(["return_pct", "value"], ascending=False)
                leaders = leaders.assign(成交額_億元=leaders.value / 100_000_000).rename(columns={"code":"代號", "name":"名稱", "market":"市場", "close":"收盤", "return_pct":"漲幅(%)"})
                leaders = add_analysis_link(leaders, "代號")
                st.dataframe(leaders[["代號", "名稱", "市場", "收盤", "漲幅(%)", "成交額_億元", "個股期", "認購權證", "認售權證", "個股分析"]].round(2), column_config=LINK_CONFIG, hide_index=True, use_container_width=True)
    st.subheader("量化結果：10 檔研究關注清單")
    st.caption("依流動性、當日動能、族群熱度與最新法人買賣超排序；僅供建立研究清單，並非買進建議。")
    try:
        watchlist = watchlist_recommendations(market).assign(成交額_億元=lambda x: x.value / 100_000_000).rename(
            columns={"code": "代號", "name": "名稱", "theme": "中文族群", "close": "最近成交價", "return_pct": "漲跌幅(%)", "quant_score": "量化分數"})
        watchlist = add_analysis_link(watchlist, "代號")
        st.dataframe(watchlist[["代號", "名稱", "中文族群", "最近成交價", "漲跌幅(%)", "個股期", "認購權證", "認售權證", "關注原因", "量化分數", "個股分析"]].round(2), column_config=LINK_CONFIG, hide_index=True, use_container_width=True)
    except requests.RequestException:
        st.info("法人資料暫時無法取得，請稍後重新整理。")

if page == "法人／月線選股":
    st.subheader("投信連買＋主力價量＋月線型態")
    st.caption("條件：上市股投信連續 3 個交易日買超；月線（20 日）斜率為正，且為月線上方強勢或強勢後拉回月線 ±3%。")
    scan_limit = st.slider("技術型態掃描檔數（依成交額與漲幅排序）", min_value=20, max_value=100, value=50, step=10)
    if st.button("執行法人／月線選股", type="primary"):
        with st.spinner("正在取得近三日投信資料與月線型態…"):
            candidates = strategy_candidates(market, scan_limit)
        if candidates.empty:
            st.info("本次沒有同時符合所有條件的股票；可提高掃描檔數，或於下一交易日再查。")
        else:
            table = candidates.assign(
                成交額_億元=candidates.value / 100_000_000,
                投信三日買超_張=candidates.trust_net_sum / 1000,
            ).rename(columns={"code": "代號", "name": "名稱", "theme": "中文族群", "setup": "技術型態",
                               "distance_to_ma20": "距月線(%)", "candidate_score": "候選分數"})
            table = table.rename(columns={"ma20_slope_5d": "月線5日斜率(%)"})
            table = table.rename(columns={"close": "最近成交價", "return_pct": "漲跌幅(%)"})
            table = add_analysis_link(table, "代號")
            st.dataframe(table[["代號", "名稱", "中文族群", "最近成交價", "漲跌幅(%)", "個股期", "認購權證", "認售權證", "技術型態", "成交額_億元", "投信三日買超_張", "月線5日斜率(%)", "距月線(%)", "候選分數", "個股分析"]].round(2), column_config=LINK_CONFIG, hide_index=True, use_container_width=True)

if page == "個股三面向":
    query = st.text_input("股票代號或名稱", value=requested_stock, placeholder="例如：2330 或 台積電")
    if query:
        try:
            code, name, symbol = resolve_symbol(query, market)
            hist = indicators(history(symbol))
            last = hist.iloc[-1]
            vals = valuation()
            val = vals[vals.code.astype(str) == code]
            val_row = val.iloc[0] if not val.empty else pd.Series({"pe": np.nan, "yield": np.nan, "pb": np.nan})
            tech, tech_notes = technical_score(last)
            fund, fund_notes = fundamental_score(val_row)
            try:
                flow_data = institutional_daily("上市" if symbol.endswith(".TW") else "上櫃")
                current_flow = flow_data[flow_data.code.astype(str) == code]
                flow = current_flow.iloc[0] if not current_flow.empty else None
            except requests.RequestException:
                flow = None
            chip, chip_notes = institutional_score(hist, flow)
            total = round(0.30 * fund + 0.45 * tech + 0.25 * chip)
            st.subheader(f"{code} {name}（{symbol}）")
            current = market[market.code.astype(str) == code].iloc[0]
            st.caption(f"題材／所屬族群：{current.theme}")
            futures_badge = "🟢 有個股期" if code in futures_underlyings else "⚪ 無個股期"
            call_badge = "🟠 有認購權證" if name in call_warrant_underlyings else "⚪ 無認購權證"
            put_badge = "🟣 有認售權證" if name in put_warrant_underlyings else "⚪ 無認售權證"
            st.markdown(f"**最近成交價：{current.close:.2f}｜漲跌幅：{current.return_pct:.2f}%｜{futures_badge}｜{call_badge}｜{put_badge}**")
            a, b, c, d = st.columns(4)
            a.metric("收盤價", f"{last.close:.2f}")
            b.metric("基本面", f"{fund}/100")
            c.metric("技術面", f"{tech}/100")
            d.metric("籌碼面", f"{chip}/100")
            st.info(f"綜合研究分數：**{total}/100 — {score_label(total)}**。請以停損、部位上限與事件風險控管為先。")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist.open, high=hist.high, low=hist.low, close=hist.close, name="股價"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.ma20, name="MA20"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.ma60, name="MA60"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.bb_upper, name="布林上軌", line=dict(color="#ff9f43", dash="dot")))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.bb_lower, name="布林下軌", line=dict(color="#ff9f43", dash="dot"), fill="tonexty", fillcolor="rgba(255,159,67,0.08)"))
            fig.update_layout(height=450, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
            left, right = st.columns(2)
            with left:
                st.markdown("**基本面與技術面**")
                st.write("・" + "\n・".join(fund_notes + tech_notes))
                st.caption(f"PE：{val_row.pe if pd.notna(val_row.pe) else '—'}｜殖利率：{val_row['yield'] if pd.notna(val_row['yield']) else '—'}%｜PB：{val_row.pb if pd.notna(val_row.pb) else '—'}")
            with right:
                st.markdown("**籌碼面**")
                st.write("・" + "\n・".join(chip_notes))
                if flow is None:
                    st.warning("籌碼分數目前為公開價量代理；不可將其解讀為三大法人買賣超。")
            st.markdown("**近 10 個交易日三大法人買賣超（股）**")
            if symbol.endswith(".TW"):
                with st.spinner("正在取得近 10 日法人明細…"):
                    institution_history = twse_institutional_history(code)
                if institution_history.empty:
                    st.info("暫時無法取得法人歷史明細。")
                else:
                    st.dataframe(institution_history, hide_index=True, use_container_width=True)
            else:
                st.info("上櫃近 10 日法人明細需逐日 TPEx 分點／法人歷史資料介接；目前先顯示最新日籌碼資料。")
        except Exception as exc:
            st.warning(str(exc))

if page == "模型說明與風險":
    st.markdown("""
    **每日掃描**：官方產業代號轉為中文，並以成交額（45%）、平均漲幅（35%）、上漲家數比（20%）形成族群熱度分數；被動元件等常用次族群另行覆寫。  
    **個股分數**：基本面 30%（PE、殖利率、PB）、技術面 45%（均線、RSI、量能）、籌碼面 25%（官方三大法人買賣超；不可得時才標示並改用價量代理）。

    **法人／月線選股**：投信連買使用 TWSE 每日公開明細；「主力」以流動性與正向價量參與代理，並非券商分點或大戶資料。

    本工具適合用於建立候選清單與紀律化複盤；不會預測未來報酬。期貨槓桿會放大損益，應另外依保證金、最大可承受損失、流動性及合約到期日制定風控規則。
    """)
