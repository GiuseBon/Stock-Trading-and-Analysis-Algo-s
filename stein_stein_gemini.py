import numpy as np
from scipy import integrate

def stein_stein_S0(eta, t=1, P=1):
    delta = 1
    kappa = 1
    theta = 1
    sig0 = 1

    # Placeholder variable lambda (using eta as the integration variable)
    lmda = (eta**2 + 0.25) * t / 2 
    
    A = -delta / (kappa**2)
    B = theta * delta / (kappa**2)
    C = -lmda / ((kappa**2) * t)

    a = np.sqrt(A**2 - 2*C)
    b = -A / a

    sinh_val = np.sinh(a * (kappa**2) * t)
    cosh_val = np.cosh(a * (kappa**2) * t)

    # L parameter
    L = -A - a * (sinh_val + b * cosh_val) / (cosh_val + b * sinh_val)
    
    # M parameter (Added the missing - 1)
    M = B * (((b * sinh_val + (b**2) * cosh_val + 1 - (b**2)) / (cosh_val + b * sinh_val)) - 1)
    
    # N parameter (Fixed the 2*A - a sign error)
    term1 = ((a - A) / (2 * a**2)) * (a**2 - A * (B**2) - (B**2) * a) * (kappa**2) * t
    term2 = ((B**2) * (A**2 - a**2) / (2 * a**3)) * ((2*A + a + (2*A - a) * np.exp(2 * a * kappa**2 * t)) / (A + a + (a - A) * np.exp(2 * a * kappa**2 * t)))
    term3 = (2 * A * (B**2) * (a**2 - A**2) * np.exp(a * kappa**2 * t)) / (a**3 * (A + a + (a - A) * np.exp(2 * a * kappa**2 * t)))
    term4 = -0.5 * np.log(0.5 * (A / a + 1) + 0.5 * (1 - A / a) * np.exp(2 * a * kappa**2 * t))
    
    N = term1 + term2 + term3 + term4
    I = np.exp(L * sig0**2 / 2 + M * sig0 + N)

    # Integrand for S0 (Eq. 9)
    return ((2 * np.pi)**(-1)) * (P**(-1.5)) * I * np.exp(1j * eta * np.log(P))

# Parameters
mu = 1
t = 1
P = 1

# Eq. 10: S0 must be evaluated at P * exp(-mu * t)
effective_P = P * np.exp(-mu * t)

# Perform numerical integration for S0
result, error = integrate.quad(lambda eta: stein_stein_S0(eta, t, effective_P), -np.inf, np.inf, complex_func=True)

print("S0 integral result: ", result)
print("estimated error: ", error)

# Final S(P, t) with drift adjustment (Eq. 10)
S = np.exp(-mu * t) * result
print("S(P, t): ", S)