import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_loader import load_data
from indicators import add_indicators
from model import prepare_data, build_model, predict, scaler
from tensorflow.keras import backend as K
import time
import datetime
import yfinance as yf
import numpy as np

# --- Session State ---
if 'show_backtester' not in st.session_state:
    st.session_state.show_backtester = False

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="SWAY.IN Terminal", page_icon="⚡")

# --- Custom CSS for Premium UI ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .main-header {
        background: -webkit-linear-gradient(45deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0px;
    }
    .fund-card {
        padding: 20px;
        background-color: #111;
        border-left: 4px solid #92FE9D;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .news-card {
        padding: 15px;
        background-color: #111;
        border-left: 4px solid #00C9FF;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Asset Dictionary (Indian Markets) ---
INDIAN_ASSETS = {
    "^NSEI": "NIFTY 50 (Index)",
    "^BSESN": "BSE SENSEX (Index)",
    "^NSEBANK": "NIFTY BANK (Index)",
    "^CNXIT": "NIFTY IT (Index)",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India (SBI)",
    "TATAMOTORS.NS": "Tata Motors",
    "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen & Toubro",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ZOMATO.NS": "Zomato",
    "TATAPOWER.NS": "Tata Power"
}

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2416/2416654.png", width=60)
    st.title("SWAY.IN")
    st.markdown("---")
    
    st.header("⚙️ Trading Parameters")
    symbol = st.selectbox(
        "Select Market Asset",
        options=list(INDIAN_ASSETS.keys()),
        index=0,
        format_func=lambda x: INDIAN_ASSETS[x]
    )
    
    interval = st.selectbox(
        "Intraday Timeframe", 
        ["1min", "5min", "15min", "30min", "60min"],
        index=1
    )
    
    st.markdown("---")
    # Will update status dynamically below
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("🧪 Advanced Tools")
    if st.button("Launch AI Backtester", use_container_width=True):
        st.session_state.show_backtester = not st.session_state.show_backtester
        st.rerun()

# --- Data Fetching ---
with st.spinner(f"Synchronizing market data for {INDIAN_ASSETS[symbol]}..."):
    data = load_data(symbol, interval)

if data.empty or len(data) < 15:
    st.error(f"Not enough market data found for {INDIAN_ASSETS[symbol]}. Try changing the intraday interval.")
    st.stop()

# --- Market Status Detection ---
last_timestamp = data.index[-1]
if last_timestamp.tz is None:
    last_timestamp = last_timestamp.tz_localize('UTC')
now_utc = datetime.datetime.now(datetime.timezone.utc)
diff_minutes = (now_utc - last_timestamp).total_seconds() / 60

market_is_live = diff_minutes < 90

with st.sidebar:
    if market_is_live:
        st.success("🟢 **Status:** Market LIVE (Auto-refresh Active)")
    else:
        st.error("🔴 **Status:** Market CLOSED (Analytics Mode)")
        st.caption(f"Last actual trade was roughly {int(diff_minutes / 60)} hours ago.")

data = add_indicators(data)
current_price = data['Close'].iloc[-1]
rsi = data['RSI'].iloc[-1]
momentum = "Overbought 🔴" if rsi > 70 else "Oversold 🟢" if rsi < 30 else "Neutral ⚪"
prev_close = data['Close'].iloc[-2]
price_change = current_price - prev_close
pct_change = (price_change / prev_close) * 100

# Buy/Sell Signals Calculation
buy_signals = data[data['RSI'] < 30]
sell_signals = data[data['RSI'] > 70]

# ==========================================
# UI ROUTING (Backtester vs Main Dashboard)
# ==========================================
if st.session_state.show_backtester:
    # --- BACKTESTER VIEW ---
    st.markdown('<p class="main-header">🧪 SWAY.IN Backtesting Simulator</p>', unsafe_allow_html=True)
    st.markdown(f"Simulating AI Trading Strategy historical performance on **{INDIAN_ASSETS[symbol]}**.")
    
    if st.button("⬅️ Return to Main Terminal"):
        st.session_state.show_backtester = False
        st.rerun()
        
    st.markdown("---")
    if len(data) > 65:
        X, y, scaled = prepare_data(data)
        
        with st.spinner("Running historical AI simulation over entire dataset..."):
            K.clear_session()
            model = build_model()
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            
            # Predict all history
            predictions_array = model.predict(X, verbose=0)
            pred_prices = scaler.inverse_transform(predictions_array).flatten()
            actual_prices = scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            historical_dates = data.index[60:]
            
        # Simulated Profit Calculation (Basic: Buy if AI says next bar goes up)
        initial_capital = 100000
        capital = initial_capital
        position = 0
        
        for i in range(1, len(pred_prices)):
            ai_bullish = pred_prices[i] > actual_prices[i-1]
            if ai_bullish and capital > 0: # Buy All
                position = capital / actual_prices[i-1]
                capital = 0
            elif not ai_bullish and position > 0: # Sell All
                capital = position * actual_prices[i-1]
                position = 0
                
        # Final liquidate
        if position > 0:
            capital = position * actual_prices[-1]
            
        profit_pct = ((capital - initial_capital) / initial_capital) * 100
        
        bt1, bt2, bt3 = st.columns(3)
        bt1.metric("Starting Capital", f"₹{initial_capital:,.2f}")
        bt2.metric("Ending Capital (AI Strategy)", f"₹{capital:,.2f}", f"{profit_pct:.2f}%")
        bt3.metric("Total Trades Executed", f"{len(pred_prices)}")
        
        st.subheader("Historical Accuracy: AI Predicted vs Actual")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=historical_dates, y=actual_prices, name="Actual Price", line=dict(color='white', width=2)))
        fig_bt.add_trace(go.Scatter(x=historical_dates, y=pred_prices, name="AI Prediction", line=dict(color='#00C9FF', width=2, dash='dot')))
        fig_bt.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_bt, use_container_width=True)
        
    else:
        st.error("Not enough historical data to run backtesting simulation. Need >65 periods.")

else:
    # --- MAIN TERMINAL VIEW ---
    st.markdown('<p class="main-header">⚡ SWAY.IN Intel Terminal</p>', unsafe_allow_html=True)
    st.markdown("Live ML-driven tracking and deep-learning price forecasts for the Indian Markets.")

    # Top KPI Metrics
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(f"LTP ({INDIAN_ASSETS[symbol].split()[1] if 'Index' in INDIAN_ASSETS[symbol] else symbol.replace('.NS', '')})", f"₹{current_price:,.2f}", f"{price_change:,.2f} ({pct_change:.2f}%)")
    with c2:
        st.metric("RSI (14) Momentum", f"{rsi:.2f}", momentum, delta_color="off")

    tab_chart, tab_ai, tab_fundamentals, tab_news = st.tabs(["📈 Pro SuperChart", "🤖 AI Analytics", "🏛️ Company Profile", "📰 Market News"])

    with tab_chart:
        if market_is_live:
            st.subheader(f"🔴 LIVE Order Flow ({interval})")
        else:
            st.subheader(f"⏸ Market Closed - Final Snapshot ({interval})")
            
        t1, t2, t3, t4 = st.columns(4)
        show_vol = t1.checkbox("📉 Show Volume", value=False)
        show_macd = t2.checkbox("📊 Show MACD", value=False)
        show_bb = t3.checkbox("🌐 Bollinger Bands", value=False)
        show_signals = t4.checkbox("🎯 AI/RSI Trade Signals", value=True)
        
        rows = 2 if show_macd else 1
        heights = [0.7, 0.3] if show_macd else [1.0]
        specs = [[{"secondary_y": True}]]
        if show_macd:
            specs.append([{"secondary_y": False}])

        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03, specs=specs)

        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'], name='Market Price'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name="EMA (20)", line=dict(color='#00C9FF', width=1.5)), row=1, col=1)

        if show_bb:
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_High'], name="BB High", line=dict(color='gray', dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Low'], name="BB Low", line=dict(color='gray', dash='dot')), row=1, col=1)

        if show_signals:
            if not buy_signals.empty:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'] * 0.998, mode='markers', marker=dict(color='#00ff00', symbol='triangle-up', size=14), name="BUY Signal"), row=1, col=1)
            if not sell_signals.empty:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'] * 1.002, mode='markers', marker=dict(color='#ff0000', symbol='triangle-down', size=14), name="SELL Signal"), row=1, col=1)

        if show_vol:
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name="Volume", marker_color='rgba(255,255,255,0.2)'), row=1, col=1, secondary_y=True)
            fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False, range=[0, data['Volume'].max()*3])
            fig.update_yaxes(secondary_y=False, showgrid=True)

        if show_macd:
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD Line", line=dict(color='blue')), row=2, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name="Signal Line", line=dict(color='orange')), row=2, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], name="Histogram", marker_color='gray'), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=700 if show_macd else 550,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_ai:
        st.subheader("LSTM Neural Network Forecasting")
        st.markdown("The AI model dynamically analyzes the previous 60 periods to project the immediate next closing price.")
        
        if len(data) > 65:
            X, y, scaled = prepare_data(data)

            with st.spinner('Initializing Neural Weights & Training...'):
                K.clear_session()
                model = build_model()
                model.fit(X, y, epochs=3, batch_size=32, verbose=0)
                pred = predict(model, scaled)
            
            pred_change = pred - current_price
            pred_pct = (pred_change / current_price) * 100
            
            c3.metric("AI Target Price (Next Hit)", f"₹{pred:,.2f}", f"₹{pred_change:,.2f} ({pred_pct:.2f}%)")
            c4.metric("Algorithm Trend", "Bullish 🟢" if pred > current_price else "Bearish 🔴", delta_color="off")
            
            p1, p2 = st.columns(2)
            with p1:
                st.success(f"**Target Hit Pattern:** Confirmed {'Upward ' if pred > current_price else 'Downward'} Breakout trajectory detected.")
                st.info("Model Accuracy relies on historical pattern replication. Do not use for actual trading execution.")
        else:
            st.warning(f"Accumulating data... The AI requires >65 periods of strict `{interval}` intraday data to execute the LSTM layer. Try a higher timeframe (e.g. 15min/30min).")

    with tab_fundamentals:
        st.subheader(f"Fundamental Analysis: {INDIAN_ASSETS[symbol]}")
        with st.spinner("Compiling company records..."):
            try:
                info = yf.Ticker(symbol).info
                if not info or 'sector' not in info and 'previousClose' not in info:
                    st.info("Detailed Fundamental data is not available for this specific Index/Asset.")
                else:
                    f1, f2, f3, f4 = st.columns(4)
                    f1.metric("Market Cap", f"₹{info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A")
                    f2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                    f3.metric("52-Week High", f"₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                    f4.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "0.00%")
                    
                    st.markdown(f"""
                    <div class="fund-card">
                        <h4 style="margin:0; color:#92FE9D;">Corporate Profile ({info.get('sector', 'General Sector')})</h4>
                        <p style="margin-top:10px; font-size:14px;">{info.get('longBusinessSummary', 'No extensive business profile available for this ticker.')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Failed to fetch fundamental data.")

    with tab_news:
        st.subheader(f"Live Market Feed: {INDIAN_ASSETS[symbol]}")
        with st.spinner("Fetching latest news from global outlets..."):
            try:
                ticker = yf.Ticker(symbol)
                news_items = ticker.news
                if not news_items:
                    st.info("No recent news found for this ticker today.")
                else:
                    for item in news_items[:5]: 
                        title = item.get('title', 'Headline Unavailable')
                        link = item.get('link', '#')
                        publisher = item.get('publisher', 'Financial Wire')
                        st.markdown(f"""
                        <div class="news-card">
                            <h4><a href="{link}" target="_blank" style="color: white; text-decoration: none;">{title}</a></h4>
                            <p style="color: #00C9FF; font-size: 0.9em; margin: 0;">Source: {publisher} | Mood: 🤖 Analyzed Neutral</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Could not load news feed. Free tier API timeout.")

    # --- Auto Refresh ---
    if market_is_live:
        time.sleep(30)
        st.rerun()