import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from groq import Groq
import os
import re
import time
import json
import portfolio as pf

# 로컬(.env)과 Streamlit Cloud(secrets) 양쪽 지원
try:
    api_key = st.secrets["GROQ_API_KEY"]
except:
    with open(".env") as f:
        for line in f:
            key, val = line.strip().split("=")
            os.environ[key] = val
    api_key = os.environ["GROQ_API_KEY"]

client = Groq(api_key=api_key)

st.set_page_config(page_title="주식 AI 분석기", page_icon="📈", layout="wide")
st.title("📈 주식 AI 분석기")

@st.cache_data
def load_krx():
    """KRX CSV 로드 — KOSPI/KOSDAQ 구분해서 .KS/.KQ 접미사 부여"""
    try:
        df_krx = pd.read_csv("data_2014_20260519.csv", encoding="cp949")
        df_krx.columns = df_krx.columns.str.strip()
        name_col = [c for c in df_krx.columns if "종목명" in c][0]
        code_col = [c for c in df_krx.columns if "종목코드" in c][0]
        market_col = None
        for c in df_krx.columns:
            if "시장" in c or "구분" in c:
                market_col = c
                break

        result = {}
        for _, row in df_krx.iterrows():
            name = row[name_col]
            code = str(row[code_col]).zfill(6)
            if market_col and "KOSDAQ" in str(row[market_col]).upper():
                suffix = ".KQ"
            else:
                suffix = ".KS"
            result[name] = code + suffix
        return result
    except:
        return {}

PERIOD_MAP = {
    "1일": ("1d", "1m"),
    "5일": ("5d", "5m"),
    "1주": ("5d", "15m"),
    "3주": ("1mo", "1h"),
    "1개월": ("1mo", "1d"),
    "3개월": ("3mo", "1d"),
    "6개월": ("6mo", "1d"),
    "1년": ("1y", "1d")
}

OVERSEAS_MAP = {
    "애플": "AAPL", "apple": "AAPL",
    "테슬라": "TSLA", "tesla": "TSLA",
    "엔비디아": "NVDA", "nvidia": "NVDA",
    "마이크로소프트": "MSFT", "microsoft": "MSFT",
    "구글": "GOOGL", "google": "GOOGL", "알파벳": "GOOGL",
    "아마존": "AMZN", "amazon": "AMZN",
    "메타": "META", "meta": "META", "페이스북": "META",
    "tsmc": "TSM",
    "넷플릭스": "NFLX", "netflix": "NFLX"
}

def fetch_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        df_rt = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        return df, df_rt
    except:
        return pd.DataFrame(), pd.DataFrame()

def clean_korean(text):
    """한자, 일본어 제거"""
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    text = re.sub(r'[\u3040-\u309f\u30a0-\u30ff]+', '', text)
    return text

def ask_agent(messages, max_retry=3, temperature=0):
    """rate limit 대응 + 대화 기억"""
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                temperature=temperature,
                seed=42,
                messages=messages
            )
            return clean_korean(response.choices[0].message.content)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                time.sleep(20)
            else:
                if attempt == max_retry - 1:
                    return "⚠️ AI 응답 한도를 초과했습니다. 1분 후 다시 시도해주세요."
                time.sleep(2)
    return "⚠️ AI 응답에 실패했습니다. 잠시 후 다시 시도해주세요."

def extract_tickers_with_llm(question, krx_dict):
    """LLM으로 질문에서 종목명 추출 → KRX/해외맵에서 코드 찾기"""
    extract_prompt = f"""다음 사용자 질문에서 언급된 한국 주식 또는 해외 주식 종목명을 추출해줘.
표기가 다양해도 (영문/한글/약칭/오타) 정식 종목명으로 정규화해줘.

예시:
- "sk hynix" → "SK하이닉스"
- "하이닉스" → "SK하이닉스"
- "apple이랑 테슬라" → "애플", "테슬라"

규칙:
1. 한국 종목은 반드시 한글 정식명으로
2. 해외 종목은 한글로 (애플, 테슬라, 엔비디아 등)
3. 종목이 없으면 빈 배열
4. 반드시 JSON 형식으로만 답변, 다른 텍스트 금지

질문: {question}

답변 형식: {{"tickers": ["종목명1", "종목명2"]}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            seed=42,
            messages=[
                {"role": "system", "content": "당신은 종목명 추출기입니다. 반드시 JSON 형식으로만 답변합니다."},
                {"role": "user", "content": extract_prompt}
            ]
        )
        content = response.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            extracted = data.get("tickers", [])
        else:
            return []

        results = []
        for name in extracted:
            name_lower = name.lower().strip()
            if name_lower in OVERSEAS_MAP:
                results.append((name, OVERSEAS_MAP[name_lower]))
                continue
            if name in krx_dict:
                results.append((name, krx_dict[name]))
                continue
            for krx_name, code in krx_dict.items():
                if name in krx_name or krx_name in name:
                    results.append((krx_name, code))
                    break
        return results
    except:
        return []

krx_dict = load_krx()

# 코드로 종목명 역조회 (모의투자에서 이름 표시용)
code_to_name = {v: k for k, v in krx_dict.items()}
for k, v in OVERSEAS_MAP.items():
    if v not in code_to_name:
        code_to_name[v] = k

# 세션 상태
defaults = {
    "ticker": "005930.KS",
    "messages": [],
    "analyst_report": "",
    "search_results": {},
    "analysis_done": False,
    "current": 0,
    "rsi_val": 0,
    "chart_fig": None,
    "metrics": {},
    "search_input": "",
    "market_state": "",
    "portfolio": pf.load_portfolio()
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# 사이드바
st.sidebar.header("종목 설정")

with st.sidebar.form("search_form"):
    search_query = st.text_input("종목명 검색", value=st.session_state.search_input, placeholder="예: 삼성전자, 카카오")
    search_btn = st.form_submit_button("검색")

if search_btn and search_query:
    matched = {k: v for k, v in krx_dict.items() if search_query in k}
    if not matched:
        try:
            results = yf.Search(search_query, max_results=5)
            quotes = results.quotes
            if quotes:
                matched = {f"{q.get('shortname', q.get('longname', ''))} ({q['symbol']})": q['symbol'] for q in quotes}
        except:
            pass
    st.session_state.search_results = matched
    if not matched:
        st.sidebar.warning("검색 결과가 없습니다.")

if st.session_state.search_results:
    selected = st.sidebar.selectbox("종목 선택", list(st.session_state.search_results.keys()))
    if st.sidebar.button("이 종목으로 설정"):
        st.session_state.ticker = st.session_state.search_results[selected]
        selected_name = selected.split(" (")[0] if " (" in selected else selected
        st.session_state.search_input = selected_name
        st.session_state.messages = []
        st.session_state.analyst_report = ""
        st.session_state.analysis_done = False
        st.session_state.search_results = {}
        st.rerun()

# 종목 코드 직접 입력 (검색 실패 대비)
with st.sidebar.expander("종목 코드 직접 입력"):
    manual_code = st.text_input("코드 입력", placeholder="예: 005930.KS, AAPL")
    if st.button("이 코드로 설정"):
        if manual_code.strip():
            st.session_state.ticker = manual_code.strip()
            st.session_state.search_input = manual_code.strip()
            st.session_state.messages = []
            st.session_state.analyst_report = ""
            st.session_state.analysis_done = False
            st.rerun()

st.sidebar.markdown(f"**선택된 종목:** `{st.session_state.ticker}`")
period_label = st.sidebar.selectbox("기간", list(PERIOD_MAP.keys()), index=4)
st.sidebar.caption("⏱️ AI 분석은 보통 5~15초 걸립니다")

analyze_btn = st.sidebar.button("분석 시작")

# ===== 탭 구조 =====
tab_analysis, tab_trade, tab_portfolio = st.tabs(["📊 분석", "💰 모의투자", "📁 내 포트폴리오"])

# ============================================================
# 탭 1 — 분석
# ============================================================
with tab_analysis:
    if analyze_btn:
        ticker = st.session_state.ticker
        period, interval = PERIOD_MAP[period_label]

        with st.spinner("데이터 불러오는 중..."):
            df, df_realtime = fetch_data(ticker, period, interval)

        if df.empty:
            st.error("종목 데이터를 불러올 수 없습니다. 종목 코드나 기간을 확인해주세요.")
        else:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()

            if len(df) < 5:
                st.error(f"데이터가 부족합니다 ({len(df)}개). 더 긴 기간을 선택해주세요.")
            else:
                ma5_window = min(5, len(df) // 2)
                ma20_window = min(20, len(df) // 2)
                rsi_window = min(14, len(df) // 2)

                df["MA5"] = df["Close"].rolling(ma5_window).mean()
                df["MA20"] = df["Close"].rolling(ma20_window).mean()

                delta = df["Close"].diff()
                gain = delta.clip(lower=0).rolling(rsi_window).mean()
                loss = -delta.clip(upper=0).rolling(rsi_window).mean()
                rs = gain / loss
                df["RSI"] = 100 - (100 / (1 + rs))

                bb_window = min(20, len(df) // 2)
                df["BB_MID"] = df["Close"].rolling(bb_window).mean()
                bb_std = df["Close"].rolling(bb_window).std()
                df["BB_UPPER"] = df["BB_MID"] + 2 * bb_std
                df["BB_LOWER"] = df["BB_MID"] - 2 * bb_std

                ema12 = df["Close"].ewm(span=12, adjust=False).mean()
                ema26 = df["Close"].ewm(span=26, adjust=False).mean()
                df["MACD"] = ema12 - ema26
                df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
                df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

                latest_close = None
                market_state = "장 마감 (전일 종가 기준)"
                if not df_realtime.empty:
                    if isinstance(df_realtime.columns, pd.MultiIndex):
                        df_realtime.columns = df_realtime.columns.get_level_values(0)
                    df_realtime = df_realtime.dropna()
                    if not df_realtime.empty:
                        latest_close = float(df_realtime["Close"].iloc[-1])
                        last_time = df_realtime.index[-1]
                        now = pd.Timestamp.now(tz=last_time.tz) if last_time.tz else pd.Timestamp.now()
                        diff_min = (now - last_time).total_seconds() / 60
                        if diff_min < 30:
                            market_state = "실시간 (장중)"
                        else:
                            market_state = "장 마감 (최근 거래 기준)"

                current = latest_close if latest_close else float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2])
                change = ((current - prev) / prev) * 100

                rsi_val = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 50.0
                ma5_val = float(df["MA5"].iloc[-1]) if not pd.isna(df["MA5"].iloc[-1]) else current
                ma20_val = float(df["MA20"].iloc[-1]) if not pd.isna(df["MA20"].iloc[-1]) else current
                macd_val = float(df["MACD"].iloc[-1]) if not pd.isna(df["MACD"].iloc[-1]) else 0
                macd_sig = float(df["MACD_SIGNAL"].iloc[-1]) if not pd.isna(df["MACD_SIGNAL"].iloc[-1]) else 0
                bb_upper = float(df["BB_UPPER"].iloc[-1]) if not pd.isna(df["BB_UPPER"].iloc[-1]) else current
                bb_lower = float(df["BB_LOWER"].iloc[-1]) if not pd.isna(df["BB_LOWER"].iloc[-1]) else current

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.5, 0.25, 0.25],
                    vertical_spacing=0.12,
                    subplot_titles=("가격 + 이동평균 + 볼린저밴드", "거래량", "MACD (추세 전환 신호)")
                )

                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name="캔들",
                    increasing_line_color="red", decreasing_line_color="blue"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MA5"], name=f"MA{ma5_window}", line=dict(color="orange", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name=f"MA{ma20_window}", line=dict(color="purple", width=1)), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB 상단", line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB 하단", line=dict(color="gray", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

                colors = ["red" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "blue" for i in range(len(df))]
                fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color=colors), row=2, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue", width=1)), row=3, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="orange", width=1)), row=3, col=1)
                hist_colors = ["red" if v >= 0 else "blue" for v in df["MACD_HIST"]]
                fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Histogram", marker_color=hist_colors), row=3, col=1)

                fig.update_layout(
                    title=dict(text=f"{ticker} 종합 차트 ({period_label})", y=0.98, x=0.5, xanchor="center"),
                    xaxis_rangeslider_visible=False,
                    height=1000,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
                    margin=dict(t=120)
                )
                for ann in fig['layout']['annotations']:
                    ann['font'] = dict(size=14)

                analyst_prompt = f"""종목: {ticker}
기간: {period_label}
현재가: {current:,.0f}원
등락률: {change:+.2f}%
{ma5_window}구간 이동평균: {ma5_val:,.0f}원
{ma20_window}구간 이동평균: {ma20_val:,.0f}원
RSI: {rsi_val:.1f}
MACD: {macd_val:.2f}, Signal: {macd_sig:.2f}
볼린저밴드 상단: {bb_upper:,.0f}원, 하단: {bb_lower:,.0f}원

위 데이터를 바탕으로 현재 주가 상태를 분석해줘.
- RSI 과매수/과매도 여부
- 이동평균 골든크로스/데드크로스 여부
- MACD 신호 (MACD가 Signal보다 위면 상승신호, 아래면 하락신호)
- 볼린저밴드 위치 (현재가가 상단 근처면 과열, 하단 근처면 침체)
- 종합 추세

초보자도 이해할 수 있게 4~5문단으로 작성해. 마지막에 이 분석은 참고용임을 명시해."""

                with st.spinner("AI가 5개 지표를 종합 분석 중입니다... (5~15초)"):
                    report = ask_agent([
                        {"role": "system", "content": "당신은 주식 분석 전문가입니다. 반드시 한국어로만 답변합니다. 한자, 중국어, 일본어는 절대 사용하지 마세요."},
                        {"role": "user", "content": analyst_prompt}
                    ])

                st.session_state.analysis_done = True
                st.session_state.analyst_report = report
                st.session_state.chart_fig = fig
                st.session_state.current = current
                st.session_state.rsi_val = rsi_val
                st.session_state.market_state = market_state
                st.session_state.metrics = {
                    "current": current, "change": change,
                    "ma5": ma5_val, "ma20": ma20_val, "rsi": rsi_val,
                    "ma5_window": ma5_window, "ma20_window": ma20_window,
                    "macd": macd_val, "macd_sig": macd_sig,
                    "period_label": period_label
                }
                st.session_state.messages = []

    # 분석 결과 표시
    if st.session_state.analysis_done:
        m = st.session_state.metrics

        if "장중" in st.session_state.market_state:
            st.success(f"🟢 {st.session_state.market_state}")
        else:
            st.info(f"🔵 {st.session_state.market_state}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("현재가", f"{m['current']:,.0f}")
        col2.metric("등락률", f"{m['change']:+.2f}%")
        col3.metric(f"{m['ma5_window']}구간 이평", f"{m['ma5']:,.0f}")
        col4.metric("RSI", f"{m['rsi']:.1f}")
        macd_signal_text = "상승" if m['macd'] > m['macd_sig'] else "하락"
        col5.metric("MACD 신호", macd_signal_text)

        st.plotly_chart(st.session_state.chart_fig, use_container_width=True)

        st.subheader("🤖 AI 분석")
        st.write(st.session_state.analyst_report)

        download_text = f"""[주식 AI 분석 리포트]
종목: {st.session_state.ticker}
기간: {m['period_label']}
현재가: {m['current']:,.0f}원
등락률: {m['change']:+.2f}%
RSI: {m['rsi']:.1f}
MACD: {m['macd']:.2f} / Signal: {m['macd_sig']:.2f}

[AI 분석 내용]
{st.session_state.analyst_report}
"""
        st.download_button(
            label="📥 분석 결과 다운로드 (.txt)",
            data=download_text.encode("utf-8"),
            file_name=f"{st.session_state.ticker}_분석리포트.txt",
            mime="text/plain"
        )

        st.subheader("💬 AI에게 추가 질문하기")
        st.caption("이전 대화 내용을 기억하며, 다른 종목 비교 시 실시간 데이터를 가져옵니다")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if question := st.chat_input("궁금한 점을 물어보세요"):
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("질문 분석 중..."):
                mentioned_tickers = extract_tickers_with_llm(question, krx_dict)
            mentioned_tickers = [(n, c) for n, c in mentioned_tickers if c != st.session_state.ticker]

            extra_context = ""
            if mentioned_tickers:
                with st.spinner("비교 종목 데이터 조회 중..."):
                    for name, code in mentioned_tickers[:3]:
                        try:
                            df_other = yf.download(code, period="5d", interval="1d", auto_adjust=True, progress=False)
                            if not df_other.empty:
                                if isinstance(df_other.columns, pd.MultiIndex):
                                    df_other.columns = df_other.columns.get_level_values(0)
                                df_other = df_other.dropna()
                                other_current = float(df_other["Close"].iloc[-1])
                                other_prev = float(df_other["Close"].iloc[-2]) if len(df_other) > 1 else other_current
                                other_change = ((other_current - other_prev) / other_prev) * 100
                                unit = "원" if code.endswith((".KS", ".KQ")) else "달러"
                                extra_context += f"\n- {name} ({code}): 현재가 {other_current:,.2f}{unit}, 등락률 {other_change:+.2f}%"
                        except:
                            extra_context += f"\n- {name} ({code}): 데이터 조회 실패"

            chat_messages = [
                {"role": "system", "content": f"""당신은 주식 분석 전문가입니다. 반드시 한국어로만 답변합니다. 한자, 중국어, 일본어는 절대 사용하지 마세요.

[현재 분석 중인 종목 정보]
종목: {st.session_state.ticker}
현재가: {st.session_state.current:,.0f}원
RSI: {st.session_state.rsi_val:.1f}

[이전 AI 분석 결과]
{st.session_state.analyst_report}
{f"[질문에서 언급된 다른 종목 실시간 데이터]{extra_context}" if extra_context else ""}

중요 규칙:
1. 위에 제공된 실시간 데이터만 사용하세요.
2. 다른 종목의 주가를 추측하거나 학습 데이터에서 가져오지 마세요.
3. 데이터가 없는 종목에 대해서는 "실시간 데이터가 없어 정확한 비교가 어렵습니다"라고 답하세요."""}
            ]
            for msg in st.session_state.messages:
                chat_messages.append(msg)

            with st.spinner("답변 생성 중..."):
                answer = ask_agent(chat_messages)

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
    else:
        st.info("👈 왼쪽 사이드바에서 종목을 선택하고 '분석 시작'을 눌러주세요.")

# ============================================================
# 탭 2 — 모의투자
# ============================================================
with tab_trade:
    st.subheader("💰 모의투자")
    ticker = st.session_state.ticker
    name = code_to_name.get(ticker, ticker)

    cur_price = pf.get_current_price(ticker)
    if cur_price is None:
        st.warning("현재 선택된 종목의 가격을 불러올 수 없습니다. 사이드바에서 종목을 다시 선택해보세요.")
    else:
        st.markdown(f"**선택 종목:** {name} (`{ticker}`)")
        st.metric("현재가", f"{cur_price:,.0f}원")

        portfolio = st.session_state.portfolio
        st.info(f"보유 현금: {portfolio['cash']:,.0f}원")

        col_buy, col_sell = st.columns(2)

        with col_buy:
            st.markdown("#### 🔴 매수")
            buy_qty = st.number_input("매수 수량", min_value=1, value=1, step=1, key="buy_qty")
            st.caption(f"예상 금액: {buy_qty * cur_price:,.0f}원")
            if st.button("매수하기", key="buy_btn"):
                ok, msg = pf.buy(portfolio, ticker, name, int(buy_qty), cur_price)
                if ok:
                    pf.save_portfolio(portfolio)
                    st.session_state.portfolio = portfolio
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

        with col_sell:
            st.markdown("#### 🔵 매도")
            held = portfolio["holdings"].get(ticker, {}).get("qty", 0)
            st.caption(f"보유 수량: {held}주")
            if held > 0:
                sell_qty = st.number_input("매도 수량", min_value=1, max_value=int(held), value=1, step=1, key="sell_qty")
                st.caption(f"예상 금액: {sell_qty * cur_price:,.0f}원")
                if st.button("매도하기", key="sell_btn"):
                    ok, msg = pf.sell(portfolio, ticker, int(sell_qty), cur_price)
                    if ok:
                        pf.save_portfolio(portfolio)
                        st.session_state.portfolio = portfolio
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.caption("이 종목은 보유하고 있지 않습니다.")

# ============================================================
# 탭 3 — 포트폴리오
# ============================================================
with tab_portfolio:
    st.subheader("📁 내 포트폴리오")
    portfolio = st.session_state.portfolio

    with st.spinner("평가손익 계산 중..."):
        result = pf.evaluate(portfolio)

    col1, col2, col3 = st.columns(3)
    col1.metric("총 평가자산", f"{result['total_eval']:,.0f}원")
    col2.metric("보유 현금", f"{result['cash']:,.0f}원")
    col3.metric("총 수익률", f"{result['total_profit_rate']:+.2f}%",
                delta=f"{result['total_profit']:,.0f}원")

    if result["holdings"]:
        st.markdown("#### 보유 종목")
        rows = []
        for h in result["holdings"]:
            rows.append({
                "종목명": h["name"],
                "수량": h["qty"],
                "평균단가": f"{h['avg_price']:,.0f}",
                "현재가": f"{h['cur_price']:,.0f}",
                "평가금액": f"{h['eval_amount']:,.0f}",
                "수익률": f"{h['profit_rate']:+.2f}%"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("아직 보유 중인 종목이 없습니다. '모의투자' 탭에서 매수해보세요.")

    if portfolio["history"]:
        st.markdown("#### 거래 내역")
        hist_df = pd.DataFrame(portfolio["history"])
        hist_df = hist_df[["time", "type", "name", "qty", "price", "amount"]]
        hist_df.columns = ["시간", "구분", "종목", "수량", "단가", "금액"]
        st.dataframe(hist_df.iloc[::-1], use_container_width=True)

    st.divider()
    if st.button("⚠️ 포트폴리오 초기화 (전체 리셋)"):
        st.session_state.portfolio = pf.reset_portfolio()
        st.success("초기 자금 1,000만원으로 초기화되었습니다.")
        st.rerun()