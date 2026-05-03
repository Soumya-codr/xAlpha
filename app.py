import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import plotly.graph_objects as go
from datetime import datetime, timezone
from scipy import stats
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="BTC Forecast | AlphaI", layout="wide")

# --- CSS: CLEAN INFORMATION DENSE ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { visibility: hidden; }
    
    .block-container { padding: 2rem 4rem !important; max-width: 1400px; }

    /* Custom Typography */
    .title-main { font-size: 2.2rem; font-weight: 700; color: #ffffff; margin-bottom: 0px; letter-spacing: -0.5px; }
    .subtitle { font-size: 0.8rem; color: #8b949e; margin-bottom: 30px; }
    
    .metric-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }
    .metric-value { font-size: 1.8rem; font-weight: 600; color: #ffffff; letter-spacing: -0.5px; }
    .metric-sub { font-size: 0.75rem; color: #3fb950; margin-top: 4px; }

    .highlight-bar {
        background-color: #1c2128;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 20px 0;
        font-size: 0.9rem;
        color: #c9d1d9;
        border-left: 4px solid #f59e0b;
    }

    /* Smaller metric for row 2 */
    .metric-value-sm { font-size: 1.5rem; font-weight: 500; color: #ffffff; }

    /* Fix table styling */
    [data-testid="stDataFrame"] { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

def winkler_score(l, u, x, alpha=0.05):
    width = u - l
    if l <= x <= u:
        return width
    elif x < l:
        return width + (2 / alpha) * (l - x)
    else:
        return width + (2 / alpha) * (x - u)

# --- DATA FETCH ---
@st.cache_data(ttl=2)
def fetch_market_snapshot():
    try:
        k_url = "https://data-api.binance.vision/api/v3/klines"
        r_k = requests.get(k_url, params={"symbol": "BTCUSDT", "interval": "1h", "limit": 500}, timeout=10).json()
        df = pd.DataFrame(r_k)[[0, 4]]
        df.columns = ["time", "close"]
        df["close"] = df["close"].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        
        # Get latest price from 1m kline (most recent tick)
        t_url = "https://data-api.binance.vision/api/v3/klines"
        r_t = requests.get(t_url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1}, timeout=10).json()
        latest_price = float(r_t[0][4])  # close price of latest 1m candle
        
        return {
            "df": df, 
            "price": latest_price, 
            "timestamp": time.time()
        }
    except Exception:
        return None

# --- ORCHESTRATION ---
@st.fragment(run_every=2.0)
def render_terminal():
    state = fetch_market_snapshot()
    if not state:
        st.error("Data Sync Interrupted. Reconnecting...")
        return

    p = state["price"]
    df_p = state["df"]
    
    # 1. READ BACKTEST METRICS
    bt_cov, bt_wid, bt_win, bt_bars = 0.0, 0.0, 0.0, 0
    if os.path.exists("backtest_results.jsonl"):
        with open("backtest_results.jsonl", "r") as f:
            first_line = f.readline()
            if first_line:
                try:
                    summary = json.loads(first_line)
                    bt_cov = summary.get("coverage", 0.0)
                    bt_wid = summary.get("mean_width", 0.0)
                    bt_win = summary.get("mean_winkler", 0.0)
                    bt_bars = summary.get("bars", 0)
                except: pass

    # 2. PREDICTIVE ENGINE (Student-t)
    recent_returns = df_p["log_return"].dropna().iloc[-48:]
    vol = recent_returns.std() if not recent_returns.empty else 0.02
    drift = df_p["log_return"].dropna().iloc[-200:].mean() if len(df_p) >= 200 else 0.0001
    
    df_params = 4
    alpha = 0.05
    t_val = stats.t.ppf(1 - alpha/2, df=df_params)
    
    f_low = p * np.exp(drift - 0.5 * vol**2 - t_val * vol)
    f_high = p * np.exp(drift - 0.5 * vol**2 + t_val * vol)
    range_width = f_high - f_low
    mid_point = (f_high + f_low) / 2

    # 3. HEADER & ROW 1
    st.markdown('<div class="title-main">₿ BTC/USDT — Next-Hour Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Refreshes every 2s · GBM + Student-t · 95% confidence</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="metric-label">BTC Price</div><div class="metric-value">${p:,.2f}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-label">Predicted Low</div><div class="metric-value">${f_low:,.2f}</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-label">Predicted High</div><div class="metric-value">${f_high:,.2f}</div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-label">Range Width</div><div class="metric-value">${range_width:,.2f}</div>', unsafe_allow_html=True)
    with c5:
        delta = bt_cov - 0.95
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        st.markdown(f'<div class="metric-label">Backtest Coverage</div><div class="metric-value">{bt_cov:.3f}</div><div class="metric-sub">↑ {delta_str} vs 0.95</div>', unsafe_allow_html=True)

    st.write("")
    
    # ROW 2
    r2c1, r2c2, r2c3 = st.columns([1,1,2])
    with r2c1:
        st.markdown(f'<div class="metric-label">Avg Width (backtest)</div><div class="metric-value-sm">${bt_wid:,.2f}</div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown(f'<div class="metric-label">Mean Winkler Score</div><div class="metric-value-sm">{bt_win:,.2f}</div>', unsafe_allow_html=True)
    with r2c3:
        st.markdown(f'<div class="metric-label">Total Predictions</div><div class="metric-value-sm">{bt_bars}</div>', unsafe_allow_html=True)

    # HIGHLIGHT BAR
    st.markdown(f'<div class="highlight-bar">📊 Next-hour range: <b>{f_low:,.2f}–{f_high:,.2f}</b> · mid <b>{mid_point:,.2f}</b> · width <b>{range_width:,.2f}</b></div>', unsafe_allow_html=True)

    # 4. PERSISTENCE LOGIC
    last_t = df_p["time"].iloc[-1]
    future_t = last_t + pd.Timedelta(hours=1)
    future_t_str = future_t.isoformat()
    
    live_file = "live_predictions.jsonl"
    live_preds = []
    if os.path.exists(live_file):
        with open(live_file, "r") as f:
            for line in f:
                if line.strip():
                    try: live_preds.append(json.loads(line))
                    except: pass
                    
    if not any(pr.get("target_time") == future_t_str for pr in live_preds):
        new_pred = {
            "target_time": future_t_str,
            "made_at": datetime.now(timezone.utc).isoformat(),
            "current_price": p,
            "pred_low": f_low,
            "pred_high": f_high
        }
        with open(live_file, "a") as f:
            f.write(json.dumps(new_pred) + "\n")
        live_preds.append(new_pred)

    # 5. CHART
    df_c = df_p.iloc[-50:].copy()
    fig = go.Figure()
    
    # Historical Live Predictions Ribbon
    if live_preds:
        df_live = pd.DataFrame(live_preds)
        df_live["target_time"] = pd.to_datetime(df_live["target_time"]).dt.tz_localize(None)
        df_live = df_live[df_live["target_time"] >= df_c["time"].iloc[0]]
        if len(df_live) > 0:
            df_live = df_live.sort_values("target_time")
            fig.add_trace(go.Scatter(x=df_live["target_time"], y=df_live["pred_low"], line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_live["target_time"], y=df_live["pred_high"], fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            
    # Price Line (Orange)
    fig.add_trace(go.Scatter(x=df_c["time"], y=df_c["close"], line=dict(color="#f59e0b", width=2), mode='lines', name='Price'))
    
    # Current Forecast Band (Dotted)
    fig.add_trace(go.Scatter(x=[future_t, future_t], y=[f_low, f_high], mode='lines+markers', line=dict(color="#58a6ff", width=2, dash='dot'), name='Forecast'))
    
    fig.update_layout(
        title="<b>BTCUSDT — Last 50 Bars + Next-Hour Forecast</b><br><span style='font-size:10px;color:#8b949e;'>■ BTC Close  ■ 95% range</span>",
        template="plotly_dark", paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        margin=dict(l=0, r=40, t=50, b=0), height=350, hovermode="x unified",
        yaxis=dict(side="right", gridcolor="#161b22", zeroline=False),
        xaxis=dict(showgrid=False, zeroline=False, range=[df_c["time"].iloc[0], future_t + pd.Timedelta(hours=2)]),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 6. HISTORY TABLE & LIVE METRICS
    st.markdown("### 📋 Prediction History")
    
    history_data = []
    live_hits = 0
    live_winkler_sum = 0.0
    resolved_count = 0
    
    # Sort descending (newest first)
    for pr in sorted(live_preds, key=lambda x: x['target_time'], reverse=True):
        tt = pd.to_datetime(pr['target_time']).tz_localize(None)
        
        # Find Actual Price from df_p
        actual_row = df_p[df_p['time'] == tt]
        actual_price = actual_row['close'].values[0] if not actual_row.empty else None
        
        hit_status = "⏳ pending"
        if actual_price is not None:
            resolved_count += 1
            if pr['pred_low'] <= actual_price <= pr['pred_high']:
                hit_status = "✅"
                live_hits += 1
            else:
                hit_status = "❌"
            live_winkler_sum += winkler_score(pr['pred_low'], pr['pred_high'], actual_price, 0.05)
            
        history_data.append({
            "Time": pd.to_datetime(pr['made_at']).strftime("%Y-%m-%d %H:%M UTC"),
            "Target Hour": tt.strftime("%Y-%m-%d %H:%M UTC"),
            "Current Price": f"${pr.get('current_price', 0):,.2f}",
            "Lower 95%": f"${pr['pred_low']:,.2f}",
            "Upper 95%": f"${pr['pred_high']:,.2f}",
            "Actual": f"${actual_price:,.2f}" if actual_price else "⏳ pending",
            "Hit": hit_status
        })

    st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
    
    # Footer Metrics
    st.markdown("---")
    lc1, lc2, lc3 = st.columns(3)
    
    live_cov = (live_hits / resolved_count) if resolved_count > 0 else 0.0
    live_wink = (live_winkler_sum / resolved_count) if resolved_count > 0 else 0.0
    
    with lc1:
        st.markdown(f'<div class="metric-label">Live Coverage</div><div class="metric-value-sm">{live_cov:.3f}</div>', unsafe_allow_html=True)
    with lc2:
        st.markdown(f'<div class="metric-label">Live Winkler</div><div class="metric-value-sm">{live_wink:,.2f}</div>', unsafe_allow_html=True)
    with lc3:
        st.markdown(f'<div class="metric-label">Resolved</div><div class="metric-value-sm">{resolved_count}</div>', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.6rem; color: #8b949e; margin-top: 20px;">Data: Binance · Model: GBM + Student-t with rolling volatility</div>', unsafe_allow_html=True)

render_terminal()
