import numpy as np
from scipy.stats import norm

def _clean_sigma(sigma):
    val = float(np.max(np.array(sigma))) 
    if val > 5.0: 
        return val / 100.0
    return max(val, 0.001)

def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    try:
        S = float(S)
        K = float(K)
        T = float(max(T, 0.0001))
        r = float(r)
        
        sigma = _clean_sigma(sigma)

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_md2 = norm.cdf(-d2)

        if option_type == "call":
            delta = cdf_d1
        else:
            delta = cdf_d1 - 1

        gamma = pdf_d1 / (S * sigma * np.sqrt(T))
        vega = (S * pdf_d1 * np.sqrt(T)) / 100.0

        if option_type == "call":
            theta_raw = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
        else:
            theta_raw = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * cdf_md2
        
        theta = theta_raw / 365.0

        if option_type == "call":
            rho = (K * T * np.exp(-r * T) * cdf_d2) / 100.0
        else:
            rho = (-K * T * np.exp(-r * T) * cdf_md2) / 100.0

        return delta, gamma, theta, vega, rho

    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0

def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    try:
        T = float(max(T, 0.0001))
        sigma = _clean_sigma(sigma) 

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    except Exception:
        return 0.0

def calculate_probabilities(S, K_lower, K_upper, T, sigma, r=0.04):
    try:
        T = float(max(T, 0.0001))
        
        sigma = _clean_sigma(sigma) 
        
        vol_period = sigma * np.sqrt(T)
        
        mu_period = (r - 0.5 * sigma**2) * T

        log_S = np.log(S)
        log_K_lower = np.log(K_lower)
        log_K_upper = np.log(K_upper)

        z_lower = (log_K_lower - log_S - mu_period) / vol_period
        z_upper = (log_K_upper - log_S - mu_period) / vol_period

        prob_below = norm.cdf(z_lower)
        prob_above = 1.0 - norm.cdf(z_upper)
        
        pop = prob_below + prob_above
        
        return pop, prob_below, prob_above
    except Exception:
        return 0.0, 0.0, 0.0