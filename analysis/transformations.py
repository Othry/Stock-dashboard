import numpy as np
import pandas as pd

def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1))

def calculate_simple_returns(prices):
    return (prices / prices.shift(1)) - 1

def normalize_prices(prices, base=100):
    return (prices / prices.iloc[0]) * base

def calculate_rolling_volatility(returns, window=20):
    return returns.rolling(window=window).std()