import yfinance as yf
import pandas as pd
import streamlit as st

def get_stock_data(symbol, period="1y"):
    """
    Fetch stock data for the given symbol
    
    Args:
        symbol (str): Stock symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

def get_stock_info(symbol):
    """
    Get detailed information about a stock
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Stock information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching stock info: {e}")
        return {}

def get_market_indices():
    """
    Display market indices in the sidebar
    """
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "Russell 2000": "^RUT",
        "VIX": "^VIX"
    }
    
    for name, symbol in indices.items():
        try:
            index_data = yf.Ticker(symbol).history(period="1d")
            if not index_data.empty:
                current = index_data['Close'][-1]
                prev_close = index_data['Close'][-2] if len(index_data) > 1 else index_data['Open'][-1]
                change = current - prev_close
                change_pct = (change / prev_close) * 100
                
                st.sidebar.metric(
                    name,
                    f"{current:.2f}",
                    f"{change_pct:+.2f}%",
                    delta_color="normal" if change_pct >= 0 else "inverse"
                )
        except Exception as e:
            st.sidebar.error(f"Could not load {name}")