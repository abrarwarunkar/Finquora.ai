import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize

def black_scholes_calc(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price and Greeks
    S: Spot Price
    K: Strike Price
    T: Time to maturity (in years)
    r: Risk-free rate (annual)
    sigma: Volatility (annual)
    """
    if T <= 0 or sigma <= 0:
        return None
        
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

def optimize_portfolio(tickers, period="2y"):
    """
    Calculates the Efficient Frontier and Optimal Weights (Max Sharpe Ratio)
    """
    if not tickers:
        return None
        
    # Fetch data
    data = pd.DataFrame()
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period=period)
            if not hist.empty:
                data[t] = hist['Close']
        except:
            pass
            
    if data.empty or len(data.columns) < 2:
        return None
        
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    num_assets = len(data.columns)
    
    def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return std, returns
        
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
        p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / p_std
        
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    
    # Initialize with equal weights
    init_guess = num_assets * [1. / num_assets,]
    
    # Maximize Sharpe (minimize negative sharpe)
    optimal = minimize(negative_sharpe_ratio, init_guess, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
                        
    opt_std, opt_ret = portfolio_annualised_performance(optimal.x, mean_returns, cov_matrix)
    
    # Generate random portfolios for Efficient Frontier plot
    num_portfolios = 2000
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        p_std, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = (p_ret - 0.02) / p_std # Sharpe ratio
        
    return {
        'optimal_weights': dict(zip(data.columns, optimal.x)),
        'optimal_return': opt_ret,
        'optimal_volatility': opt_std,
        'optimal_sharpe': (opt_ret - 0.02) / opt_std,
        'frontier_volatility': results[0,:],
        'frontier_returns': results[1,:],
        'frontier_sharpe': results[2,:]
    }
