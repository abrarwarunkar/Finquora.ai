# Remove matplotlib imports and lock
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from services.stock_service import get_stock_data
from services.ml_service import predict_stock_price
from utils.ml_utils import calculate_rsi, train_model, predict_future
from sklearn.metrics import mean_squared_error, mean_absolute_error

def render_ml_anomaly_detection(ticker):
    from utils.ml_utils import detect_anomalies
    st.subheader(f"🔍 ML Anomaly Detection for {ticker}")
    st.markdown("Uses an Isolation Forest machine learning model to detect unusual trading behavior based on price and volume.")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1d")
    
    if df.empty or len(df) < 30:
        st.warning("Not enough data to run anomaly detection (need at least 30 days).")
        return
        
    with st.spinner("Training Isolation Forest and detecting anomalies..."):
        df_anomalies = detect_anomalies(df)
        
        # Plotly chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Price & Anomalies', 'Volume'))
                           
        # Normal Price vs Anomaly Price
        anomaly_mask = df_anomalies['Anomaly'] == -1
        
        fig.add_trace(go.Scatter(x=df_anomalies.index, y=df_anomalies['Close'],
                                name='Price', line=dict(color='#2962ff', width=2)), row=1, col=1)
        
        # Anomalies Price Overlay
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(x=df_anomalies[anomaly_mask].index, y=df_anomalies[anomaly_mask]['Close'],
                                    mode='markers', name='Anomaly',
                                    marker=dict(color='red', size=8, symbol='x')), row=1, col=1)
                                
        # Volume
        colors = ['red' if a == -1 else '#757575' for a in df_anomalies['Anomaly']]
        fig.add_trace(go.Bar(x=df_anomalies.index, y=df_anomalies['Volume'],
                            name='Volume', marker_color=colors), row=2, col=1)
                            
        fig.update_layout(height=600, showlegend=True, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        num_anomalies = len(df_anomalies[anomaly_mask])
        st.info(f"Detected **{num_anomalies}** unusual trading anomalies in the last 6 months.")

def render_stock_prediction(ticker):
    st.subheader(f"🔮 Price Prediction for {ticker}")
    
    # Load data first
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo", interval="1d").copy()
    
    days = st.slider("Prediction Days", min_value=1, max_value=30, value=7)
    
    if st.button("Generate Prediction"):
        with st.spinner("Generating prediction..."):
            try:
                predictions = predict_stock_price(ticker, days)  # Remove tuple unpacking
                
                if len(predictions) == 0:
                    st.warning("Unable to generate prediction. Not enough historical data available for this stock.")
                    st.info("Try a more established stock with longer trading history.")
                else:
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Actual': df['Close'][-len(predictions):],
                        'Predicted': predictions
                    })
                    
                    # Replace matplotlib plots with Plotly in the prediction section
                    if len(df) > 10:
                        # Plot actual vs predicted
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual'],
                                               name='Actual', line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted'],
                                               name='Predicted', line=dict(color='red', width=2, dash='dash')))
                        
                        fig.update_layout(
                            title='Stock Price Prediction vs Actual',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Future predictions plot
                        fig_future = go.Figure()
                        fig_future.add_trace(go.Scatter(x=df['Close'].index[-30:], y=df['Close'].values[-30:],
                                                      name='Historical', line=dict(color='blue', width=2)))
                        fig_future.add_trace(go.Scatter(x=predictions.index, y=predictions.values,
                                                      name='Predictions', line=dict(color='green', width=2)))
                        
                        # Add confidence interval
                        fig_future.add_trace(go.Scatter(
                            x=predictions.index.tolist() + predictions.index.tolist()[::-1],
                            y=(predictions.values * 1.05).tolist() + (predictions.values * 0.95).tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,128,0,0.1)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval'
                        ))
                        
                        fig_future.update_layout(
                            title='Price Forecast',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_future, use_container_width=True)
                    
                    # Display prediction table
                    st.subheader("Predicted Prices")
                    pred_df = pd.DataFrame({
                        'Date': predictions.index,
                        'Predicted Price': [f"${price:.2f}" for price in predictions.values]
                    })
                    st.dataframe(pred_df)
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.info("Try using a different stock symbol or reducing the number of prediction days.")
    
    stock = yf.Ticker(ticker)
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Use Current Stock Data", "Upload Custom CSV"],
        help="Choose whether to use current stock data or upload your own CSV file"
    )
    
    if data_source == "Use Current Stock Data":
        df = stock.history(period="2y", interval="1d").copy()
    else:
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="CSV must contain columns: Date, Open, High, Low, Close, Volume"
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Date' not in df.columns:
                    st.error("CSV must contain a 'Date' column")
                    return
                
                # Convert Date column to datetime
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_columns):
                    st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                    return
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return
        else:
            st.info("Please upload a CSV file to continue")
            return
    
    # Continue with prediction if enough data
    if len(df) > 30:
        try:
            # Feature engineering
            df['MA7'] = df['Close'].rolling(window=7).mean()
            df['MA21'] = df['Close'].rolling(window=21).mean()
            df['RSI'] = calculate_rsi(df['Close'])
            df['Target'] = df['Close'].shift(-1)
            
            # Remove NaN values
            df.dropna(inplace=True)
            
            # Check if we still have enough data after dropna
            if len(df) < 30:
                st.warning("Not enough valid data points after feature engineering.")
                return
                
            # Split features and target
            X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'RSI']]
            y = df['Target']
            
            # Train-test split
            split_point = int(len(df) * 0.8)
            X_train = X[:split_point]
            X_test = X[split_point:]
            y_train = y[:split_point]
            y_test = y[split_point:]
            
            # Model training and prediction
            with st.spinner("Training prediction model..."):
                model, scaler = train_model(X_train, y_train)
                
                # Scale test data and make predictions
                X_test_scaled = scaler.transform(X_test)
                predictions = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with col2:
                    st.metric("MAE", f"${mae:.2f}")
                with col3:
                    st.metric("Accuracy Score", f"{model.score(X_test_scaled, y_test):.2%}")
        except Exception as e:
            st.error(f"Error during ML training: {str(e)}")
            return
            
            # Combine actual and predicted data
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions
            }, index=y_test.index)
            
            # Plot predictions vs actual using Plotly
            fig_actual = go.Figure()
            fig_actual.add_trace(go.Scatter(
                x=results_df.index, y=results_df['Actual'],
                name='Actual', line=dict(color='blue', width=2)
            ))
            fig_actual.add_trace(go.Scatter(
                x=results_df.index, y=results_df['Predicted'],
                name='Predicted', line=dict(color='red', width=2, dash='dash')
            ))
            fig_actual.update_layout(
                title='Stock Price Prediction vs Actual',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_actual, use_container_width=True)
            
            # Future predictions
            days_to_predict = st.slider("Days to predict", 1, 30, 7, key="days_to_predict_slider")
            future_pred = predict_future(model, scaler, X, days_to_predict)
            
            # Show future predictions
            st.subheader("Future Price Predictions")
            
            # Create combined Plotly chart
            fig_future2 = go.Figure()
            fig_future2.add_trace(go.Scatter(
                x=df['Close'].index[-30:], y=df['Close'].values[-30:],
                name='Historical', line=dict(color='blue', width=2)
            ))
            if not future_pred.empty:
                fig_future2.add_trace(go.Scatter(
                    x=future_pred.index, y=future_pred.values,
                    name='Predictions', line=dict(color='green', width=2)
                ))
                fig_future2.add_trace(go.Scatter(
                    x=future_pred.index.tolist() + future_pred.index.tolist()[::-1],
                    y=(future_pred.values * 1.05).tolist() + (future_pred.values * 0.95).tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,128,0,0.1)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
            fig_future2.update_layout(
                title='Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_future2, use_container_width=True)

def render_ai_sentiment_analysis(ticker):
    from services.groq_service import analyze_news_sentiment
    st.subheader(f"🧠 AI Sentiment Analysis for {ticker}")
    st.markdown("Fetches real-time news articles and uses the Groq LLM to analyze market sentiment.")
    
    stock = yf.Ticker(ticker)
    
    with st.spinner("Fetching news and analyzing sentiment..."):
        try:
            news = stock.news
            if not news:
                st.warning("No recent news found for this ticker.")
                return
                
            sentiment_analysis = analyze_news_sentiment(news, ticker)
            
            st.markdown("### 📊 AI Analysis Result")
            st.info(sentiment_analysis)
            
            with st.expander("View Source Articles"):
                for item in news[:5]:
                    content = item.get('content')
                    if content and isinstance(content, dict):
                        title = content.get('title') or 'No Title'
                        link_info = content.get('clickThroughUrl') or {}
                        link = link_info.get('url') if isinstance(link_info, dict) else '#'
                    else:
                        title = item.get('title') or 'No Title'
                        link = item.get('link') or '#'
                    st.write(f"- [{title}]({link})")
                    
        except Exception as e:
            st.error(f"Error during sentiment analysis: {str(e)}")