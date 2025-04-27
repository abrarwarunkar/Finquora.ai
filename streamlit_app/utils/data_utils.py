import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_agent_response(response: str) -> str:
    """Process the response from the Gemini AI agent"""
    # Gemini responses are already well-formatted, but we can add some
    # additional processing if needed
    
    # Add markdown formatting for better readability
    processed_response = response
    
    # Highlight financial terms with bold
    import re
    financial_terms = [
        "stock", "bond", "dividend", "yield", "market cap", "P/E ratio",
        "bull market", "bear market", "volatility", "ETF", "mutual fund",
        "portfolio", "asset allocation", "diversification", "liquidity"
    ]
    
    for term in financial_terms:
        # Use word boundaries to match whole words/phrases only
        pattern = r'\b' + re.escape(term) + r'\b'
        processed_response = re.sub(
            pattern, 
            f"**{term}**", 
            processed_response, 
            flags=re.IGNORECASE
        )
    
    return processed_response

def parse_date(date_str):
    """Parse date string to datetime object"""
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def calculate_returns(prices):
    """Calculate returns from price series"""
    return prices.pct_change().dropna()

def calculate_volatility(returns, window=30):
    """Calculate rolling volatility"""
    return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
    return np.sqrt(252) * excess_returns.mean() / returns.std()

def calculate_drawdown(prices):
    """Calculate drawdown from price series"""
    rolling_max = prices.cummax()
    drawdown = (prices / rolling_max) - 1
    return drawdown

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown from price series"""
    drawdown = calculate_drawdown(prices)
    return drawdown.min()

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def format_large_number(value):
    """Format large numbers with K, M, B suffixes"""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"