import numpy as np
from scipy.integrate import quad

def compute_I(eta, t, delta, theta, k, sigma_0):
    """Computes the moment generating function I(lambda) with overflow protection."""
    lam = (eta**2 + 0.25) * (t / 2.0)
    
    A = -delta / (k**2)
    B = (theta * delta) / (k**2)
    C = -lam / (k**2 * t)
    
    val_inside_sqrt = A**2 - 2 * C
    if val_inside_sqrt < 0:
        return 0.0
    
    a = np.sqrt(val_inside_sqrt)
    b = -A / a
    
    ak2t = np.clip(a * (k**2) * t, -50.0, 50.0)
    sinh_val = np.sinh(ak2t)
    cosh_val = np.cosh(ak2t)
    denominator = cosh_val + b * sinh_val
    
    if abs(denominator) < 1e-12:
        return 0.0
        
    L = -A - a * (sinh_val + b * cosh_val) / denominator
    M = B * ((b * sinh_val + (b**2) * cosh_val + 1 - b**2) / denominator - 1)
    
    exp_2ak2t = np.exp(np.clip(2 * ak2t, -50.0, 50.0))
    exp_ak2t = np.exp(np.clip(ak2t, -50.0, 50.0))
    
    term1 = ((a - A) / (2 * a**2)) * (a**2 - A * (B**2) - (B**2) * a) * (k**2) * t
    term2 = ((B**2) * (A**2 - a**2) / (2 * a**3)) * ((2 * A + a) + (2 * A - a) * exp_2ak2t) / (A + a + (a - A) * exp_2ak2t)
    term3 = (2 * A * (B**2) * (a**2 - A**2) * exp_ak2t) / (a**3 * (A + a + (a - A) * exp_2ak2t))
    
    log_arg = 0.5 * (A / a + 1) + 0.5 * (1 - A / a) * exp_2ak2t
    if log_arg <= 0:
        return 0.0
    term4 = -0.5 * np.log(log_arg)
    
    N = term1 + term2 + term3 + term4
    exponent = L * (sigma_0**2) / 2.0 + M * sigma_0 + N
    
    if exponent > 50.0:
        return 0.0
        
    return np.exp(exponent)

def S_0_normalized(y, t, delta, theta, k, sigma_0, eta_limit=1000.0):
    """Computes S_0(y, t) where initial stock price is normalized to 1.0 (y = P / P_0)."""
    x = np.log(y)
    
    def integrand(eta):
        I_val = compute_I(eta, t, delta, theta, k, sigma_0)
        return I_val * np.cos(eta * x)

    integral, _ = quad(integrand, 0.0, eta_limit, limit=300)
    integral *= 2.0  # Symmetry adjustment
    
    S0_val = (1.0 / (2.0 * np.pi)) * (y ** (-1.5)) * integral
    return S0_val

def stock_price_distribution_normalized(y, t, mu, delta, theta, k, sigma_0):
    """Computes general normalized distribution S(y, t) for drift mu."""
    scaled_y = y * np.exp(-mu * t)
    S0_val = S_0_normalized(scaled_y, t, delta, theta, k, sigma_0)
    return np.exp(-mu * t) * S0_val

def european_call_price(K, P_0, t, r, delta, theta, k, sigma_0, phi=0.0):
    """
    Computes European call option price by working in normalized price units (y = P / P_0).
    """
    theta_hat = theta - (phi * k) / delta if delta != 0 else theta
    mu = r
    
    k_norm = K / P_0
    
    def payoff_integrand(y):
        pdf = stock_price_distribution_normalized(y, t, mu, delta, theta_hat, k, sigma_0)
        return (y - k_norm) * pdf

    upper_bound = max(4.0, k_norm * 3.0)
    integral_val, _ = quad(payoff_integrand, k_norm, upper_bound, limit=200)
    
    # Scale back by P_0 and discount by riskless rate
    call_price = P_0 * np.exp(-r * t) * integral_val
    return call_price

if __name__ == "__main__":
    P_0 = 100.0
    K = 100.0         # At-the-money option strike
    t = 0.25          # 3-month maturity
    r = 0.0953        # Riskless rate (approx ln(1.1))
    
    # Parameters from Panel A[cite: 11]
    delta = 4.0
    theta = 0.20
    k = 0.10
    sigma_0 = 0.20
    phi = 0.0         # Zero risk premium[cite: 11]
    
    price = european_call_price(K, P_0, t, r, delta, theta, k, sigma_0, phi)
    print(f"Stochastic Volatility European Call Price (K={K}, t={t}): {price:.4f}")