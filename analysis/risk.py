import numpy as np
import pandas as pd


def to_scalar(val):
    if isinstance(val, (pd.Series, pd.DataFrame, np.ndarray)):
        try:
            if isinstance(val, pd.DataFrame):
                return float(val.iloc[0, 0])
            if len(val) > 0:
                return float(val.iloc[0]) if hasattr(val, 'iloc') else float(val[0])
            return 0.0
        except:
            return 0.0
    return float(val)


def calculate_annualized_volatility(log_returns):
    return log_returns.std() * np.sqrt(252)


def calculate_value_at_risk(log_returns, confidence_level=0.95):
    if log_returns.empty: return 0.0
    return np.percentile(log_returns, (1 - confidence_level) * 100)


def calculate_max_drawdown(prices):
    prices = prices.dropna()
    if prices.empty: return 0.0
    
    running_max = prices.cummax()
    drawdown = (prices - running_max) / running_max
    return drawdown.min()


def get_normal_distribution_curve(returns):
    cleaned_returns = returns.dropna()
    
    if cleaned_returns.empty:
        return np.array([]), np.array([])
    
    mean = cleaned_returns.mean()
    sigma = cleaned_returns.std()
    
    mean = to_scalar(mean)
    sigma = to_scalar(sigma)

    if sigma == 0 or np.isnan(sigma):
        return np.array([]), np.array([])

    x = np.linspace(cleaned_returns.min(), cleaned_returns.max(), 100)
    
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma)**2)
    
    return x, pdf


def calculate_rolling_volatility(returns, window_size=30):
    rolling_std = returns.rolling(window=window_size).std()
    return rolling_std * np.sqrt(252)
