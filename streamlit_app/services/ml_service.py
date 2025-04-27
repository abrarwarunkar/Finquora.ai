import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import yfinance as yf
from utils.ml_utils import calculate_rsi, train_model, predict_future

def predict_stock_price(ticker, days=7):
    """
    Predict stock price for the given ticker
    
    Args:
        ticker (str): Stock symbol
        days (int): Number of days to predict
        
    Returns:
        pandas.Series: Predicted prices
    """
    try:
        # Fetch more data to ensure enough samples for rolling features
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")  # Get maximum available history
        
        print(f"Initial data shape for {ticker}: {df.shape}")
        
        if len(df) < 100:  # Increased minimum requirement
            print(f"Not enough historical data for {ticker}. Found only {len(df)} days, need at least 100.")
            return pd.Series([], name="Prediction")
        
        # Handle missing values in all columns
        for column in df.columns:
            if df[column].isna().any():
                if column == 'Volume':
                    # Fill Volume with median to avoid skewing the data
                    df[column] = df[column].fillna(df[column].median())
                else:
                    # For price columns, use forward fill then backward fill
                    df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
        
        # Feature engineering with more robust error handling
        try:
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            df['Target'] = df['Close'].shift(-1)
        except Exception as e:
            print(f"Error during feature engineering: {e}")
            return pd.Series([], name="Prediction")
        
        # Remove NaN values
        df_clean = df.dropna().copy()
        
        print(f"Data shape after cleaning for {ticker}: {df_clean.shape}")
        
        # Ensure we have enough data after cleaning
        if len(df_clean) < 30:
            print(f"Not enough clean data for {ticker}. Found only {len(df_clean)} samples after preprocessing.")
            return pd.Series([], name="Prediction")
        
        # Use only the most recent data for training
        if len(df_clean) > 1000:
            df_clean = df_clean.tail(1000)
            print(f"Using the most recent 1000 data points for training.")
        
        # Split features and target
        X = df_clean[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'RSI']]
        y = df_clean['Target']
        
        # Train model
        model, scaler = train_model(X, y)
        
        # Predict future prices
        future_pred = predict_future(model, scaler, X, days)
        
        return future_pred
    except Exception as e:
        import traceback
        print(f"Error predicting stock price: {e}")
        print(traceback.format_exc())
        return pd.Series([], name="Prediction")

def run_financial_agent(prompt):
    """
    Run the financial agent with the given prompt
    
    Args:
        prompt (str): User prompt
        
    Returns:
        str: Agent response
    """
    try:
        from financial_agent import run
        response = run(prompt)
        return response
    except Exception as e:
        print(f"Error running financial agent: {e}")
        return f"I'm sorry, I couldn't process your request: {str(e)}"