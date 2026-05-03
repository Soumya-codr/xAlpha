import numpy as np

def simulate_gbm(S0, mu, sigma, n_sims=10000, verbose=True):
    """
    Simulates next-step prices using a Geometric Brownian Motion (GBM) 
    approach with Student-t distributed shocks.
    
    Parameters:
    - S0: Initial price
    - mu: Expected return (drift)
    - sigma: Volatility
    - n_sims: Number of simulations
    - verbose: Whether to print statistics
    
    Returns:
    - An array of simulated prices
    """
    # Student-t distribution with degrees of freedom = 5
    df_t = 5
    Z = np.random.standard_t(df_t, size=n_sims)
    
    # Compute next-step prices using the GBM formula:
    # S_next = S0 * exp((mu - 0.5 * sigma^2) + sigma * Z)
    exponent = (mu - 0.5 * sigma**2) + sigma * Z
    S_next = S0 * np.exp(exponent)
    
    # Ensure no negative or zero prices (exp is always positive, but as a safety measure)
    S_next = np.clip(S_next, a_min=1e-10, a_max=None)
    
    if verbose:
        # Print requested statistics
        print(f"Shape of the output: {S_next.shape}")
        print(f"Min of simulated prices: {S_next.min():.4f}")
        print(f"Max of simulated prices: {S_next.max():.4f}")
    
    return S_next

if __name__ == "__main__":
    # Example values (can be linked to real data in future steps)
    S0_val = 78070.73
    mu_val = 0.0001
    sigma_val = 0.01
    
    prices = simulate_gbm(S0_val, mu_val, sigma_val)
