import streamlit as st
import yfinance as yf
import pandas as pd
from services.stock_service import get_stock_data

# Add this to your render_dashboard function

def render_dashboard(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo", interval="1d")
    if hist.empty:
        st.error(f"No data available for {ticker}")
        return
        
    st.subheader(stock.info.get("longName", ticker))
    
    # Moving average selector
    ma_window = st.selectbox(
        "Moving Average Window", 
        [7, 14, 30, 50, 200], 
        index=1,
        help="Select the window size for moving average calculation"
    )
    hist['MA'] = hist['Close'].rolling(window=ma_window).mean()
    
    # Enhanced chart
    st.line_chart(hist[['Close', 'MA']].rename(columns={
        'Close': 'Price',
        'MA': f'{ma_window}-Day MA'
    }), use_container_width=True)
    
    # Calculate percentage change
    change = hist['Close'][-1] - hist['Close'][-2]
    pct_change = (change / hist['Close'][-2]) * 100
    
    # Metrics layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price", 
            f"${hist['Close'][-1]:.2f}",
            f"{pct_change:+.2f}%",
            help="Most recent closing price"
        )
    with col2:
        st.metric(
            "52-Week High", 
            f"${stock.info.get('fiftyTwoWeekHigh', 'N/A')}",
            f"${(stock.info.get('fiftyTwoWeekHigh', 0) - hist['Close'][-1]):.2f} from current",
            help="Highest price in the last 52 weeks"
        )
    with col3:
        st.metric(
            "Trading Volume", 
            f"{stock.info.get('volume', 0):,.0f}",
            f"{((stock.info.get('volume', 0) / stock.info.get('averageVolume', 1)) - 1)*100:+.1f}% vs avg",
            help="Today's trading volume"
        )
        st.metric(
            "52-Week Low", 
            f"${stock.info.get('fiftyTwoWeekLow', 'N/A')}",
            help="Lowest price in the last 52 weeks"
        )
    
    # Fundamentals expander
    with st.expander("📈 Fundamental Metrics", expanded=False):
        fundamentals = {
            "Valuation": {
                "P/E Ratio": stock.info.get('trailingPE', 'N/A'),
                "PEG Ratio": stock.info.get('pegRatio', 'N/A'),
                "Price/Book": stock.info.get('priceToBook', 'N/A'),
            },
            "Profitability": {
                "Profit Margin": f"{stock.info.get('profitMargins', 0)*100:.2f}%" if stock.info.get('profitMargins') else 'N/A',
                "ROE": f"{stock.info.get('returnOnEquity', 0)*100:.2f}%" if stock.info.get('returnOnEquity') else 'N/A',
            },
            "Risk": {
                "Beta": stock.info.get('beta', 'N/A'),
                "Short % Float": f"{stock.info.get('shortPercentOfFloat', 0)*100:.2f}%" if stock.info.get('shortPercentOfFloat') else 'N/A',
            }
        }
        
        for category, metrics in fundamentals.items():
            st.markdown(f"**{category}**")
            cols = st.columns(len(metrics))
            for (metric, value), col in zip(metrics.items(), cols):
                with col:
                    st.metric(metric, value)