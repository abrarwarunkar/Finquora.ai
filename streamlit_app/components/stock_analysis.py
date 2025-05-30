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

def render_stock_analysis(ticker):
    # Add app name and tagline at the top
    st.title("Finquora")
    st.caption("Smart Financial Insights for Every Step")
    
    st.subheader("📈 Technical Indicators")
    
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo", interval="1d")
    if hist.empty:
        st.error(f"No data available for {ticker}")
        return
    
    # Calculate technical indicators
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    ma_window = 14
    hist['MA'] = hist['Close'].rolling(window=ma_window).mean()
    
    # Create technical analysis chart using Plotly
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Price & MA', 'RSI', 'MACD'))

    # Price and MA plot
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'],
                            name='Price', line=dict(color='#2962ff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MA'],
                            name=f'{ma_window}-Day MA', line=dict(color='#00e676', dash='dash')), row=1, col=1)

    # RSI plot
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi,
                            name='RSI', line=dict(color='#2962ff')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD plot
    fig.add_trace(go.Scatter(x=macd.index, y=macd,
                            name='MACD', line=dict(color='#2962ff')), row=3, col=1)
    fig.add_trace(go.Scatter(x=signal.index, y=signal,
                            name='Signal', line=dict(color='#00e676')), row=3, col=1)
    
    # MACD histogram
    colors = ['green' if val > 0 else 'red' for val in (macd - signal)]
    fig.add_trace(go.Bar(x=macd.index, y=macd-signal,
                        name='MACD Histogram',
                        marker_color=colors), row=3, col=1)

    fig.update_layout(height=800, showlegend=True,
                     title_text="Technical Analysis")
    
    st.plotly_chart(fig, use_container_width=True)

    # Indicator interpretation
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RSI (14)", f"{rsi[-1]:.1f}")
        if rsi[-1] > 70:
            st.error("Overbought (>70) - Potential pullback expected")
        elif rsi[-1] < 30:
            st.success("Oversold (<30) - Potential rebound expected")
        else:
            st.info("Neutral (30-70) - No strong signal")
    
    with col2:
        macd_status = "Bullish" if macd[-1] > signal[-1] else "Bearish"
        st.metric("MACD Signal", macd_status)
        st.caption(f"MACD: {macd[-1]:.2f}")
        st.caption(f"Signal Line: {signal[-1]:.2f}")
        if (macd[-1] > 0 and signal[-1] > 0):
            st.success("Positive Territory - Bullish")
        elif (macd[-1] < 0 and signal[-1] < 0):
            st.error("Negative Territory - Bearish")

def render_corporate_calendar(ticker):
    stock = yf.Ticker(ticker)
    
    st.subheader("📅 Earnings Calendar")
    try:
        earnings = stock.calendar
        if earnings and isinstance(earnings, dict):
            # Convert dictionary to DataFrame
            earnings_df = pd.DataFrame([earnings])
            
            # Format the earnings data safely
            formatted_data = {}
            for col in earnings_df.columns:
                if 'Date' in col:
                    # Handle datetime columns
                    formatted_data[col] = earnings_df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) 
                        else str(x) if pd.notnull(x) else 'N/A'
                    )
                elif any(x in col for x in ['Average', 'Low', 'High']):
                    # Handle numeric columns
                    formatted_data[col] = earnings_df[col].apply(
                        lambda x: f"${x:.2f}" if pd.notnull(x) else 'N/A'
                    )
                else:
                    formatted_data[col] = earnings_df[col]
            
            # Display formatted earnings data
            st.dataframe(pd.DataFrame(formatted_data), use_container_width=True)
            
            # Display next earnings date if available
            if 'Earnings Date' in earnings_df.columns and not earnings_df['Earnings Date'].empty:
                next_earnings = earnings_df['Earnings Date'].iloc[0]
                if isinstance(next_earnings, (datetime, pd.Timestamp)):
                    days_until = (next_earnings - datetime.now()).days
                    st.metric(
                        "Next Earnings Date", 
                        next_earnings.strftime('%b %d, %Y'), 
                        f"{days_until} days" if days_until > 0 else "Today"
                    )
        else:
            st.info("No upcoming earnings events found for this security.")
    except Exception as e:
        st.error(f"Could not fetch earnings data: {str(e)}")
    
    st.subheader("💰 Dividend Information")
    try:
        dividends = stock.dividends
        if not dividends.empty:
            last_dividend = dividends[-1]
            last_ex_date = dividends.index[-1]
            
            cols = st.columns(2)
            with cols[0]:
                st.metric(
                    "Last Dividend", 
                    f"${last_dividend:.2f}",
                    help=f"Paid on {last_ex_date.strftime('%b %d, %Y')}"
                )
            with cols[1]:
                if stock.info.get('dividendYield'):
                    st.metric(
                        "Dividend Yield", 
                        f"{stock.info['dividendYield']*100:.2f}%",
                        help="Annual dividend yield based on current price"
                    )
            
            # Dividend chart
            st.area_chart(
                dividends.tail(12), 
                use_container_width=True
            )
        else:
            st.info("This security does not currently pay dividends.")
    except Exception as e:
        st.error(f"Could not fetch dividend data: {str(e)}")

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
        df = stock.history(period="1mo", interval="1d").copy()
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
    if len(df) > 10:
        # Feature engineering
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Target'] = df['Close'].shift(-1)
        
        # Remove NaN values
        df.dropna(inplace=True)
        
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
            
            # Plot predictions vs actual
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Combine actual and predicted data
            results_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': predictions
            }, index=y_test.index)
            
            # Plot with styling
            ax.plot(results_df.index, results_df['Actual'], 
                   label='Actual', color='blue', linewidth=2)
            ax.plot(results_df.index, results_df['Predicted'], 
                   label='Predicted', color='red', linestyle='--', linewidth=2)
            
            ax.set_title('Stock Price Prediction vs Actual', fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Future predictions
            days_to_predict = st.slider("Days to predict", 1, 30, 7)
            future_pred = predict_future(model, scaler, X, days_to_predict)
            
            # Show future predictions
            st.subheader("Future Price Predictions")
            
            # Create combined plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical data
            ax.plot(df['Close'].index[-30:], df['Close'].values[-30:], 
                   label='Historical', color='blue', linewidth=2)
            
            # Plot future predictions
            ax.plot(future_pred.index, future_pred.values, 
                   label='Predictions', color='green', linewidth=2)
            
            ax.axvline(x=df.index[-1], color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(future_pred.index, 
                          future_pred.values * 0.95, 
                          future_pred.values * 1.05, 
                          color='green', alpha=0.1)
            
            ax.set_title('Price Forecast', fontsize=14, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

def render_news_feed(ticker):
    stock = yf.Ticker(ticker)
    
    st.subheader("📰 Latest News")
    try:
        # Fetch news for the selected stock
        news = stock.news
        
        for article in news[:10]:  # Display top 10 news items
            with st.expander(article['title']):
                st.write(f"**Source:** {article['source']}")
                st.write(f"**Published:** {datetime.fromtimestamp(article['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(article['summary'])
                st.markdown(f"[Read More]({article['link']})")
    except Exception as e:
        st.error(f"Could not fetch news: {str(e)}")