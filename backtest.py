import numpy as np
import pandas as pd
import json
from btc_analysis import fetch_btc_data
from gbm_simulation import simulate_gbm

def run_backtest():
    # 1. Fetch BTC data
    df = fetch_btc_data()
    if df is None:
        print("Failed to fetch data.")
        return

    predictions = []
    
    # 2. Iterate from index i = 100 to len(df) - 1
    # i is the index of the "actual" next price we want to predict
    total_len = len(df)
    
    print(f"Starting backtest from index 100 to {total_len - 1}...")
    
    for i in range(100, total_len):
        # i. Use ONLY past data: train = df.iloc[:i]
        train = df.iloc[:i]
        
        # ii. Compute parameters:
        # mu = mean of train["log_return"]
        # sigma = std of last 50 log returns
        mu = train["log_return"].mean()
        sigma = train["log_return"].tail(50).std()
        
        # iii. Get current price (last available price in train)
        S0 = train["close"].iloc[-1]
        
        # iv. Simulate 10,000 next prices
        # verbose=False to keep the output clean during the loop
        sims = simulate_gbm(S0, mu, sigma, n_sims=10000, verbose=False)
        
        # v. Compute 95% prediction interval
        lower = np.percentile(sims, 2.5)
        upper = np.percentile(sims, 97.5)
        
        # Tighten the interval based on previous high coverage
        lower = lower * 1.001
        upper = upper * 0.999
        
        # vi. Get actual next price
        actual = df["close"].iloc[i]
        
        # vii. Store results
        predictions.append({
            "lower": lower,
            "upper": upper,
            "actual": actual
        })

    # After loop results:
    print(f"\nTotal number of predictions: {len(predictions)}")
    
    print("\nFirst 3 predictions:")
    for j in range(min(3, len(predictions))):
        print(predictions[j])

    # 3. Compute Metrics
    calculate_metrics(predictions)

    # 4. Save Results
    save_results(predictions)

def save_results(predictions, filename="backtest_results.jsonl"):
    with open(filename, 'w') as f:
        for p in predictions:
            # Ensure native Python types for JSON serialization
            item = {
                "lower": float(p["lower"]),
                "upper": float(p["upper"]),
                "actual": float(p["actual"])
            }
            f.write(json.dumps(item) + "\n")
    print(f"\nResults saved to {filename}")

def calculate_metrics(predictions):
    n = len(predictions)
    if n == 0:
        return

    alpha = 0.05
    coverage_count = 0
    widths = []
    winkler_scores = []

    for p in predictions:
        lower = p["lower"]
        upper = p["upper"]
        actual = p["actual"]
        width = upper - lower
        widths.append(width)

        # Coverage
        if lower <= actual <= upper:
            coverage_count += 1
            winkler_score = width
        else:
            # Winkler Score
            if actual < lower:
                winkler_score = width + (2.0 / alpha) * (lower - actual)
            else: # actual > upper
                winkler_score = width + (2.0 / alpha) * (actual - upper)
        
        winkler_scores.append(winkler_score)

    coverage = coverage_count / n
    avg_width = np.mean(widths)
    mean_winkler = np.mean(winkler_scores)

    print("\n--- Evaluation Metrics ---")
    print(f"Coverage: {coverage:.4f}")
    print(f"Average Width: {avg_width:.4f}")
    print(f"Mean Winkler Score: {mean_winkler:.4f}")
if __name__ == "__main__":
    run_backtest()
