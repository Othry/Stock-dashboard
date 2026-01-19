import pandas as pd
import numpy as np
from scipy import optimize

def convert_to_portfolio_currency(price_series, exchange_rate_series):
    if exchange_rate_series is None:
        return price_series
    
    df = pd.concat([price_series, exchange_rate_series], axis=1).ffill().dropna()
    df.columns = ['price', 'fx']
    
    return df['price'] * df['fx']

def calculate_buy_and_hold_series(price_df, shares_dict):
    portfolio_value = pd.Series(0.0, index=price_df.index)
    
    for ticker, shares in shares_dict.items():
        if ticker in price_df.columns:
            portfolio_value += price_df[ticker] * shares
            
    portfolio_returns = np.log(portfolio_value / portfolio_value.shift(1).replace(0, np.nan))
    
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    return portfolio_value, portfolio_returns

def calculate_annualized_returns(log_returns):
    if len(log_returns) < 1: return 0.0
    return log_returns.mean() * 252

def calculate_annualized_volatility(log_returns):
    if len(log_returns) < 1: return 0.0
    return log_returns.std() * np.sqrt(252)

def calculate_sharpe_ratio(log_returns, risk_free_rate):
    ret_ann = calculate_annualized_returns(log_returns)
    vol_ann = calculate_annualized_volatility(log_returns)
    
    if vol_ann == 0: return 0.0
    return (ret_ann - risk_free_rate) / vol_ann

def calculate_beta(asset_returns, benchmark_returns):
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2: return 0.0
    
    a_rets = asset_returns.loc[common_idx]
    b_rets = benchmark_returns.loc[common_idx]
    
    cov_matrix = np.cov(a_rets, b_rets)
    cov = cov_matrix[0, 1]
    var_b = cov_matrix[1, 1]
    
    if var_b == 0: return 0.0
    return cov / var_b

def calculate_correlation(asset_returns, benchmark_returns):
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 2: return 0.0
    return asset_returns.loc[common_idx].corr(benchmark_returns.loc[common_idx])


def calculate_cagr(price_series):
    if len(price_series) < 2: return 0.0
    
    start_val = price_series.iloc[0]
    end_val = price_series.iloc[-1]
    
    if start_val <= 0: return 0.0
    
    years = len(price_series) / 252.0
    
    if years == 0: return 0.0
    
    return (end_val / start_val) ** (1 / years) - 1


def calculate_total_return(current_value, total_invested):
    if total_invested == 0: return 0.0
    return (current_value / total_invested) - 1

def calculate_xirr(cashflows):
    if not cashflows: return 0.0
    
    dates = [cf[0] for cf in cashflows]
    amounts = [cf[1] for cf in cashflows]
    
    if len(dates) < 2: return 0.0
    
    start_date = min(dates)
    
    def xnpv(rate, amounts, dates):
        if rate <= -1.0: return float('inf')
        val = 0.0
        for i, amount in enumerate(amounts):
            days_passed = (dates[i] - start_date).days
            val += amount / ((1 + rate) ** (days_passed / 365.0))
        return val

    try:
        return optimize.newton(lambda r: xnpv(r, amounts, dates), 0.1, maxiter=100)
    except:
        return 0.0