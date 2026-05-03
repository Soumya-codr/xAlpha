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
st.set_page_config(page_title="AlphaI | BTC Forecast Terminal", layout="wide", initial_sidebar_state="collapsed")

# --- CSS: PREMIUM GLASSMORPHIC DESIGN ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #0a0e1a 0%, #0d1320 40%, #111827 100%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stHeader"], [data-testid="stToolbar"], footer, #MainMenu,
    [data-testid="stSidebar"] { visibility: hidden; display: none; }
    
    .block-container { padding: 1.8rem 3.5rem !important; max-width: 1500px; }

    /* ── GLASS CARD ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 12px;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover { border-color: rgba(255, 255, 255, 0.12); }

    /* ── ACCENT CARDS (with colored top border) ── */
    .accent-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }
    .accent-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-from), var(--accent-to));
        border-radius: 16px 16px 0 0;
    }
    .accent-cyan { --accent-from: #06b6d4; --accent-to: #22d3ee; }
    .accent-emerald { --accent-from: #10b981; --accent-to: #34d399; }
    .accent-amber { --accent-from: #f59e0b; --accent-to: #fbbf24; }
    .accent-violet { --accent-from: #8b5cf6; --accent-to: #a78bfa; }
    .accent-rose { --accent-from: #f43f5e; --accent-to: #fb7185; }

    /* ── TYPOGRAPHY ── */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .hero-sub {
        font-size: 0.78rem;
        color: #64748b;
        letter-spacing: 0.5px;
        margin-bottom: 28px;
    }
    .hero-sub span {
        display: inline-block;
        background: rgba(16, 185, 129, 0.1);
        color: #34d399;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
    }

    .metric-label {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.5px;
    }
    .metric-value-sm {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .metric-delta {
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-up { color: #34d399; }
    .delta-down { color: #fb7185; }

    /* ── HIGHLIGHT BAR ── */
    .forecast-bar {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(34, 211, 238, 0.04) 100%);
        border: 1px solid rgba(6, 182, 212, 0.15);
        border-radius: 12px;
        padding: 14px 20px;
        margin: 18px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .forecast-bar b { color: #22d3ee; }

    /* ── SECTION HEADER ── */
    .section-header {
        font-size: 1rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.3px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-header .badge {
        font-size: 0.6rem;
        background: rgba(139, 92, 246, 0.15);
        color: #a78bfa;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        letter-spacing: 1px;
    }

    /* ── FOOTER ── */
    .footer-text {
        font-size: 0.6rem;
        color: #475569;
        margin-top: 24px;
        letter-spacing: 0.5px;
    }

    /* ── TABLE STYLING ── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
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
        
        t_url = "https://data-api.binance.vision/api/v3/klines"
        r_t = requests.get(t_url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1}, timeout=10).json()
        latest_price = float(r_t[0][4])
        
        return {
            "df": df, 
            "price": latest_price, 
            "timestamp": time.time()
        }
    except Exception:
        return None

# --- MAIN RENDER ---
@st.fragment(run_every=2.0)
def render_terminal():
    state = fetch_market_snapshot()
    if not state:
        st.markdown('<div class="glass-card" style="text-align:center; padding:60px;"><div style="font-size:1.2rem; color:#fb7185; font-weight:600;">⚠ Data Sync Interrupted</div><div style="color:#64748b; font-size:0.8rem; margin-top:8px;">Reconnecting to Binance...</div></div>', unsafe_allow_html=True)
        return

    p = state["price"]
    df_p = state["df"]
    
    # ── BACKTEST METRICS ──
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

    # ── PREDICTION ENGINE ──
    recent_returns = df_p["log_return"].dropna().iloc[-48:]
    vol = recent_returns.std() if not recent_returns.empty else 0.02
    drift = df_p["log_return"].dropna().iloc[-200:].mean() if len(df_p) >= 200 else 0.0001
    
    df_params = 5
    alpha = 0.05
    t_val = stats.t.ppf(1 - alpha/2, df=df_params)
    
    f_low = p * np.exp(drift - 0.5 * vol**2 - t_val * vol)
    f_high = p * np.exp(drift - 0.5 * vol**2 + t_val * vol)
    range_width = f_high - f_low
    mid_point = (f_high + f_low) / 2

    # ═══════════════════════════════════════════
    # ── HERO HEADER ──
    # ═══════════════════════════════════════════
    st.markdown('<div class="hero-title">₿ BTC/USDT — Next-Hour Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">GBM + Student-t · 95% confidence · Refreshes every 2s<span>● LIVE</span></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # ── ROW 1: PRIMARY METRICS (Glass Cards) ──
    # ═══════════════════════════════════════════
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="accent-card accent-cyan"><div class="metric-label">BTC Price</div><div class="metric-value">${p:,.2f}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="accent-card accent-emerald"><div class="metric-label">Predicted Low</div><div class="metric-value">${f_low:,.2f}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="accent-card accent-emerald"><div class="metric-label">Predicted High</div><div class="metric-value">${f_high:,.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="accent-card accent-amber"><div class="metric-label">Range Width</div><div class="metric-value">${range_width:,.2f}</div></div>', unsafe_allow_html=True)
    with c5:
        delta = bt_cov - 0.95
        delta_cls = "delta-up" if delta >= 0 else "delta-down"
        delta_sign = "+" if delta >= 0 else ""
        st.markdown(f'<div class="accent-card accent-violet"><div class="metric-label">Backtest Coverage</div><div class="metric-value">{bt_cov:.3f}</div><div class="metric-delta {delta_cls}">↑ {delta_sign}{delta:.3f} vs 0.95</div></div>', unsafe_allow_html=True)

    st.write("")

    # ═══════════════════════════════════════════
    # ── ROW 2: SECONDARY METRICS ──
    # ═══════════════════════════════════════════
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Avg Width (Backtest)</div><div class="metric-value-sm">${bt_wid:,.2f}</div></div>', unsafe_allow_html=True)
    with r2:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Mean Winkler Score</div><div class="metric-value-sm">{bt_win:,.2f}</div></div>', unsafe_allow_html=True)
    with r3:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Total Predictions</div><div class="metric-value-sm">{bt_bars}</div></div>', unsafe_allow_html=True)

    # ── FORECAST BAR ──
    st.markdown(f'<div class="forecast-bar">📊 Next-hour range: <b>${f_low:,.2f} – ${f_high:,.2f}</b> &nbsp;·&nbsp; mid <b>${mid_point:,.2f}</b> &nbsp;·&nbsp; width <b>${range_width:,.2f}</b></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # ── PERSISTENCE LOGIC ──
    # ═══════════════════════════════════════════
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

    # ═══════════════════════════════════════════
    # ── CHART ──
    # ═══════════════════════════════════════════
    df_c = df_p.iloc[-50:].copy()
    fig = go.Figure()
    
    # Prediction ribbon
    if live_preds:
        df_live = pd.DataFrame(live_preds)
        df_live["target_time"] = pd.to_datetime(df_live["target_time"]).dt.tz_localize(None)
        df_live = df_live[df_live["target_time"] >= df_c["time"].iloc[0]]
        if len(df_live) > 0:
            df_live = df_live.sort_values("target_time")
            fig.add_trace(go.Scatter(x=df_live["target_time"], y=df_live["pred_low"], line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=df_live["target_time"], y=df_live["pred_high"], fill='tonexty', fillcolor='rgba(6, 182, 212, 0.06)', line=dict(width=0), showlegend=False, hoverinfo='skip'))
            
    # Price line (cyan)
    fig.add_trace(go.Scatter(
        x=df_c["time"], y=df_c["close"],
        line=dict(color="#06b6d4", width=2.5, shape='spline'),
        mode='lines', name='BTC Close'
    ))
    
    # Forecast marker
    fig.add_trace(go.Scatter(
        x=[future_t, future_t], y=[f_low, f_high],
        mode='lines+markers',
        line=dict(color="#a78bfa", width=2, dash='dot'),
        marker=dict(size=8, color="#a78bfa", symbol="diamond"),
        name='95% Forecast'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=50, t=10, b=0), height=380,
        hovermode="x unified",
        yaxis=dict(side="right", gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=10, color="#64748b", family="JetBrains Mono")),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#64748b", family="JetBrains Mono"), range=[df_c["time"].iloc[0], future_t + pd.Timedelta(hours=2)]),
        showlegend=False,
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_color="#e2e8f0")
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ═══════════════════════════════════════════
    # ── PREDICTION HISTORY TABLE ──
    # ═══════════════════════════════════════════
    st.markdown('<div class="section-header">📋 Prediction History <span class="badge">PART C</span></div>', unsafe_allow_html=True)
    
    history_data = []
    live_hits = 0
    live_winkler_sum = 0.0
    resolved_count = 0
    
    for pr in sorted(live_preds, key=lambda x: x['target_time'], reverse=True):
        tt = pd.to_datetime(pr['target_time']).tz_localize(None)
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

    # Backtest rows (last 10)
    if os.path.exists("backtest_results.jsonl"):
        bt_rows = []
        with open("backtest_results.jsonl", "r") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        if not row.get("summary"):
                            bt_rows.append(row)
                    except: pass
        # Use last 11 rows so we can get "current price" from previous row
        recent_bt = bt_rows[-11:]
        for i in range(1, len(recent_bt)):
            r = recent_bt[i]
            prev_price = recent_bt[i-1]["actual"]  # price model saw when predicting
            history_data.append({
                "Time": "backtest",
                "Target Hour": r["timestamp"],
                "Current Price": f"${prev_price:,.2f}",
                "Lower 95%": f"${r['pred_low']:,.2f}",
                "Upper 95%": f"${r['pred_high']:,.2f}",
                "Actual": f"${r['actual']:,.2f}",
                "Hit": "✅" if r["covered"] else "❌"
            })

    if history_data:
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, hide_index=True)
    else:
        st.info("Collecting predictions... Data will populate as hours pass.")

    # ═══════════════════════════════════════════
    # ── FOOTER LIVE METRICS ──
    # ═══════════════════════════════════════════
    # Include backtest resolved data in footer metrics
    total_resolved = resolved_count
    total_hits = live_hits
    total_winkler = live_winkler_sum
    
    if os.path.exists("backtest_results.jsonl"):
        with open("backtest_results.jsonl", "r") as f:
            first_line = f.readline()
            if first_line:
                try:
                    s = json.loads(first_line)
                    bt_n = s.get("bars", 0)
                    total_resolved += bt_n
                    total_hits += int(s.get("coverage", 0) * bt_n)
                    total_winkler += s.get("mean_winkler", 0) * bt_n
                except: pass

    overall_cov = (total_hits / total_resolved) if total_resolved > 0 else 0.0
    overall_wink = (total_winkler / total_resolved) if total_resolved > 0 else 0.0

    st.markdown("---")
    lc1, lc2, lc3 = st.columns(3)
    
    with lc1:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Live Coverage</div><div class="metric-value-sm">{overall_cov:.3f}</div></div>', unsafe_allow_html=True)
    with lc2:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Live Winkler</div><div class="metric-value-sm">{overall_wink:,.2f}</div></div>', unsafe_allow_html=True)
    with lc3:
        st.markdown(f'<div class="glass-card"><div class="metric-label">Resolved</div><div class="metric-value-sm">{total_resolved}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="footer-text">Data: Binance (data-api.binance.vision) · Model: GBM + Student-t with rolling volatility · Built by AlphaI</div>', unsafe_allow_html=True)

render_terminal()
