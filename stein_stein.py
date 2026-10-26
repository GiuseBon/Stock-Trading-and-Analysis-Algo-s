import math
from scipy import integrate
import numpy as np
def stein_stein(x):
    delta = 1
    kappa = 1
    theta = 1
    sig0 = 1
    t = 1
    P = 1

    #stein and stein 1991 :Stock Price Distributions with Stochastic Volatility: An Analytic Approach closed form formula

    lmda = (x**2 + 1/4)*t/2 #placeholder variable
    

    A = (-1)*delta/(kappa**2)

    B = theta*delta/(kappa**2)

    C = (-1)*lmda/((kappa**2) * t)

    a = (A**2 - 2*C)**(1/2)

    b = (-1)*A/a

    L = -A - a*(np.sinh(a*(kappa**2)*t) + (b)*np.cosh(a*(kappa**2) * t)) /(np.cosh(a*(kappa**2)*t) + b*np.sinh(a*(kappa**2)*t))

    M = B*((b*np.sinh(a*(kappa**2)*t) + (b**2)*np.cosh(a*(kappa**2) * t) + 1 - (b**2))/(np.cosh(a*(kappa**2)*t) + b*np.sinh(a*(kappa**2)*t)) - 1)

    N = (((a-A)/(2*a*a))*(a**2 - A*(B**2) - B*B*a)*(kappa**2)*t  
         +  (B*B*(A*A - a*a)/(2*a*a*a))*((2*A + a + (2*A - a)*np.exp(2*a*kappa*kappa*t))/(A + a +(a - A)*np.exp(2*a*kappa*kappa*t))) 
         + ((2*A*B*B*(a*a - A*A)*np.exp(a*kappa*kappa*t))/((a**3)*(A + a + (a-A)*np.exp(2*a*kappa*kappa*t)))) 
         - (1/2)*np.log((1/2)*(A/a + 1) + (1/2)*(1-A/a)*np.exp(2*a*kappa*kappa*t))
         )
    I = np.exp(L*sig0*sig0/2 + M*sig0 + N)

    #S0(P,t), we need to solve for the integral using numerical methods

    #set lmda to (n^2 + 1/4)*t/2 we will take the moment generating function and I integral from negative inf to positive inf
    return ((2*np.pi)**(-1))*(P**(-3/2))*I*np.exp(1j*x*np.log(P))

result, error = integrate.quad(stein_stein,-100,100,complex_func=True) #oveflow errors when the bounds are too high due to sinh functions not going to zero but going exponential.
mu = 1
t = 1
P = 1

print("integral result: ",result)
print("estimated error: ",error)

S = float(np.exp(-mu*t)*result*(P*np.exp(-mu*t)))
print("S: ",S)