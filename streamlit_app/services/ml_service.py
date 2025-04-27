import pandas as pd
import numpy as np
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
        pandas.Series: Predicted prices with datetime index
    """
    try:
        # Fetch historical data
        stock = yf.Ticker(ticker)
        df = stock.history(period="max")
        
        if len(df) < 100:
            print(f"Not enough historical data for {ticker}. Found only {len(df)} days.")
            return pd.Series([], name="Prediction")
        
        # Handle missing values
        for column in df.columns:
            if df[column].isna().any():
                if column == 'Volume':
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(method='ffill').fillna(method='bfill')
        
        # Feature engineering
        try:
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            df['Target'] = df['Close'].shift(-1)
        except Exception as e:
            print(f"Error during feature engineering: {e}")
            return pd.Series([], name="Prediction")
        
        df_clean = df.dropna().copy()
        
        if len(df_clean) < 30:
            print(f"Not enough clean data for {ticker}.")
            return pd.Series([], name="Prediction")
        
        # Use recent data for better predictions
        if len(df_clean) > 1000:
            df_clean = df_clean.tail(1000)
        
        # Prepare features
        X = df_clean[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'RSI']]
        y = df_clean['Target']
        
        # Train and predict
        model, scaler = train_model(X, y)
        future_dates = pd.date_range(start=df.index[-1], periods=days+1)[1:]
        future_pred = predict_future(model, scaler, X, days)
        future_pred.index = future_dates
        
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