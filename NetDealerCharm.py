import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

def calculate_charm(S, K, T, r, q, sigma, option_type='call'):
    """
    Calculates Black-Scholes Charm (annual delta decay rate).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_neg_d1 = norm.cdf(-d1)

    term1 = pdf_d1 * ((2 * (r - q) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))

    if option_type.lower() == 'call':
        return -np.exp(-q * T) * (term1 - q * cdf_d1)
    else:
        return -np.exp(-q * T) * (term1 + q * cdf_neg_d1)


def get_net_dealer_charm_exposure(
    ticker_symbol: str, 
    risk_free_rate: float = 0.045, 
    dividend_yield: float = 0.0,
    days_in_year: float = 365.0
):
    """
    Calculates Net Dealer Charm Exposure across the full option chain.
    """
    ticker = yf.Ticker(ticker_symbol)
    
    # Fetch spot price
    history = ticker.history(period="1d")
    if history.empty:
        raise ValueError(f"Could not retrieve stock data for {ticker_symbol}")
    spot_price = history["Close"].iloc[-1]
    
    expirations = ticker.options
    if not expirations:
        raise ValueError(f"No option expiration dates found for {ticker_symbol}")

    all_options = []
    today = pd.Timestamp.now().normalize()

    for exp_date in expirations:
        t_days = (pd.Timestamp(exp_date) - today).days
        if t_days <= 0:
            continue
        T = t_days / 365.0

        opt_chain = ticker.option_chain(exp_date)

        for option_type, df in [('call', opt_chain.calls), ('put', opt_chain.puts)]:
            for _, row in df.iterrows():
                strike = row['strike']
                iv = row['impliedVolatility']
                oi = row['openInterest']

                if pd.isna(iv) or iv <= 0 or pd.isna(oi) or oi <= 0:
                    continue

                charm_annual = calculate_charm(
                    S=spot_price,
                    K=strike,
                    T=T,
                    r=risk_free_rate,
                    q=dividend_yield,
                    sigma=iv,
                    option_type=option_type
                )
                
                # Convert annual charm to daily charm
                charm_daily = charm_annual / days_in_year

                # Apply dealer positioning sign:
                # Customers buy calls (dealer short call -> -1)
                # Customers buy puts (dealer long put -> +1)
                dealer_sign = -1.0 if option_type == 'call' else 1.0

                all_options.append({
                    'expiration': exp_date,
                    'type': option_type,
                    'strike': strike,
                    'open_interest': oi,
                    'implied_volatility': iv,
                    'charm_daily': charm_daily,
                    'dealer_sign': dealer_sign,
                    # Contract exposure = Daily Charm * Sign * 100 shares
                    'dealer_charm_per_contract': charm_daily * dealer_sign * 100
                })

    df_chain = pd.DataFrame(all_options)

    # -------------------------------------------------------------------------
    # Dot Product / Vector Calculation
    # -------------------------------------------------------------------------
    # Vector of (Daily Charm per share * Dealer Sign * 100 shares/contract)
    dealer_charm_vector = df_chain['charm_daily'].to_numpy() * df_chain['dealer_sign'].to_numpy() * 100
    oi_vector = df_chain['open_interest'].to_numpy()

    # Net Dealer Share Delta Exposure per Day
    net_dealer_share_delta_day = np.dot(dealer_charm_vector, oi_vector)

    # Net Dealer Dollar Delta Exposure per Day ($ Exposure)
    net_dealer_dollar_delta_day = net_dealer_share_delta_day * spot_price

    return df_chain, spot_price, net_dealer_share_delta_day, net_dealer_dollar_delta_day


if __name__ == "__main__":
    ticker_name = "SOFI"
    df_chain, spot, share_delta_exposure, dollar_delta_exposure = get_net_dealer_charm_exposure(ticker_name)

    print(f"Ticker: {ticker_name}")
    print(f"Current Spot Price: ${spot:.2f}")
    print(f"Total Evaluated Options: {len(df_chain)}")
    print("--------------------------------------------------")
    print(f"Net Dealer Daily Delta Decay (Shares): {share_delta_exposure:,.2f} deltas/day")
    print(f"Net Dealer Daily Delta Decay (Dollars): ${dollar_delta_exposure:,.2f} / day")