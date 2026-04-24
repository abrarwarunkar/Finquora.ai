import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    # Handle edge cases
    if len(prices) <= window:
        # Not enough data for RSI calculation
        return pd.Series(np.nan, index=prices.index)
    
    delta = prices.diff()
    
    # Handle division by zero
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    
    # Replace zeros in loss with a small number to avoid division by zero
    loss = loss.replace(0, 1e-10)
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(X, y):
    """
    Train a machine learning model for stock price prediction
    
    Args:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        
    Returns:
        tuple: (model, scaler)
    """
    # Check if we have enough data
    if len(X) < 30:
        raise ValueError(f"Not enough data for training. Found only {len(X)} samples, need at least 30.")
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_future(model, scaler, X, days=7):
    """
    Predict future stock prices
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X (pandas.DataFrame): Features
        days (int): Number of days to predict
        
    Returns:
        pandas.Series: Predicted prices
    """
    try:
        # Get the last available data point
        last_date = X.index[-1]
        last_price = X['Close'].iloc[-1]
        
        # Create date range for predictions
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
        predictions = []
        
        # Make a copy of the last row to avoid modifying the original data
        current_features = X.iloc[-1:].copy().values
        
        # Ensure we have the right shape
        if current_features.shape[1] != 8:  # Expected columns: Open, High, Low, Close, Volume, MA7, MA21, RSI
            print(f"Unexpected feature shape: {current_features.shape}")
            return pd.Series([], name="Prediction")
        
        for _ in range(days):
            try:
                # Scale the features
                scaled_features = scaler.transform(current_features)
                
                # Make prediction
                pred = model.predict(scaled_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simple approach)
                # In a real implementation, you would need a more sophisticated approach
                current_features[0, 0] = pred  # Update Open with previous close
                current_features[0, 1] = pred * 1.01  # Simulate High as 1% above prediction
                current_features[0, 2] = pred * 0.99  # Simulate Low as 1% below prediction
                current_features[0, 3] = pred  # Update Close price
                # Keep Volume the same
                # Update MA7 and MA21 (simple approximation)
                current_features[0, 5] = (current_features[0, 5] * 6 + pred) / 7  # Update MA7
                current_features[0, 6] = (current_features[0, 6] * 20 + pred) / 21  # Update MA21
                # Keep RSI the same for simplicity
            except Exception as e:
                print(f"Error during prediction iteration: {e}")
                break
        
        if not predictions:
            return pd.Series([], name="Prediction")
            
        return pd.Series(predictions, index=future_dates, name="Prediction")
    except Exception as e:
        print(f"Error in predict_future: {e}")
        return pd.Series([], name="Prediction")

def detect_anomalies(df):
    """
    Detect anomalies in stock data using Isolation Forest
    
    Args:
        df (pandas.DataFrame): Stock history dataframe
        
    Returns:
        pandas.DataFrame: DataFrame with an 'Anomaly' column (-1 for anomaly, 1 for normal)
    """
    from sklearn.ensemble import IsolationForest
    try:
        # We need enough data
        if len(df) < 30:
            return df
            
        features = ['Close', 'Volume']
        
        # Calculate daily returns to use as a feature
        df_analysis = df.copy()
        df_analysis['Returns'] = df_analysis['Close'].pct_change()
        df_analysis.dropna(inplace=True)
        
        # Use returns and volume for anomaly detection
        X = df_analysis[['Returns', 'Volume']].values
        
        # Scale features manually or just pass to IsolationForest
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model
        # contamination sets the expected proportion of outliers (e.g. 5%)
        model = IsolationForest(contamination=0.05, random_state=42)
        df_analysis['Anomaly'] = model.fit_predict(X_scaled)
        
        # Rejoin with original df
        df['Anomaly'] = 1 # default to normal
        df.loc[df_analysis.index, 'Anomaly'] = df_analysis['Anomaly']
        
        return df
    except Exception as e:
        print(f"Error detecting anomalies: {e}")
        df['Anomaly'] = 1
        return df