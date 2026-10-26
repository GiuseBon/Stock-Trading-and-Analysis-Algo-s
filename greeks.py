import numpy as np
from scipy.stats import norm
from scipy.optimize import newton

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calculate_greeks(S, K, T, r, market_price, option_type='call'):
    # 1. Solve for Implied Volatility (IV)
    # Objective function: BS_Price(sigma) - Market_Price = 0
    func = lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    
    try:
        iv = newton(func, x0=0.2) # Initial guess of 20% vol
    except:
        return "Could not converge to an IV"

    # 2. Calculate d1 using the found IV
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    
    # 3. Calculate Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # 4. Calculate Gamma
    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T))
    
    return {"IV": iv, "Delta": delta, "Gamma": gamma}

# Example Usage:
# S=100, K=100, T=0.5 (6 months), r=0.05, Price=10.0
results = calculate_greeks(100, 100, 0.5, 0.05, 10.0, 'call')
print(results)