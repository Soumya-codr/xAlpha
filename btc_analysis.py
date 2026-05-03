import requests
import pandas as pd
import numpy as np
from scipy import stats
import json

def fetch_btc_data(limit=1000):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame(data)[[0, 4]]
    df.columns = ['time', 'close']
    df['close'] = df['close'].astype(float)
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.sort_values('time').reset_index(drop=True)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df

def winkler_score(l, u, x, alpha=0.05):
    width = u - l
    if l <= x <= u:
        return width
    elif x < l:
        return width + (2 / alpha) * (l - x)
    else:
        return width + (2 / alpha) * (x - u)

def run_backtest():
    print("Fetching data...")
    df = fetch_btc_data(1000)
    df = df.dropna().reset_index(drop=True)
    
    backtest_length = 720
    if len(df) <= backtest_length:
        raise ValueError("Not enough data to run 720 bar backtest with warmup.")
    
    start_idx = len(df) - backtest_length
    results = []
    
    df_params = 4 # Student-t degrees of freedom for fat tails
    alpha = 0.05
    t_val = stats.t.ppf(1 - alpha/2, df=df_params)
    
    print(f"Starting walk-forward backtest for {backtest_length} bars...")
    
    for i in range(start_idx, len(df)):
        # STRICT NO PEEKING: Only use data up to i-1
        hist = df.iloc[:i]
        
        # Volatility Clustering: Use a rolling window of recent volatility
        # e.g., standard deviation of last 48 hours
        recent_returns = hist['log_return'].iloc[-48:]
        vol = recent_returns.std()
        
        # Drift: using a slightly longer window or zero
        drift = hist['log_return'].iloc[-200:].mean()
        
        p = hist['close'].iloc[-1]
        
        # Student-t interval prediction
        f_low = p * np.exp(drift - 0.5 * vol**2 - t_val * vol)
        f_high = p * np.exp(drift - 0.5 * vol**2 + t_val * vol)
        
        # Actual outcome
        actual = df['close'].iloc[i]
        actual_time = df['time'].iloc[i]
        
        # Evaluation
        is_covered = f_low <= actual <= f_high
        width = f_high - f_low
        w_score = winkler_score(f_low, f_high, actual, alpha)
        
        results.append({
            "timestamp": actual_time.isoformat(),
            "pred_low": f_low,
            "pred_high": f_high,
            "actual": actual,
            "covered": bool(is_covered),
            "width": width,
            "winkler": w_score
        })
        
    # Summarize
    coverage = np.mean([r['covered'] for r in results])
    mean_width = np.mean([r['width'] for r in results])
    mean_winkler = np.mean([r['winkler'] for r in results])
    
    print(f"--- BACKTEST RESULTS ---")
    print(f"Coverage 95%   : {coverage:.4f} (Target ~0.95)")
    print(f"Mean Width     : ${mean_width:,.2f}")
    print(f"Mean Winkler   : {mean_winkler:.2f}")
    
    # Save to JSONL
    out_file = "backtest_results.jsonl"
    with open(out_file, "w") as f:
        # Write header with overall metrics as first line for easy app parsing
        f.write(json.dumps({
            "summary": True,
            "coverage": coverage,
            "mean_width": mean_width,
            "mean_winkler": mean_winkler,
            "bars": backtest_length
        }) + "\n")
        
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Saved predictions and metrics to {out_file}")

if __name__ == "__main__":
    run_backtest()
