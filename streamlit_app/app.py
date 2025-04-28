import streamlit as st
from dotenv import load_dotenv
import os
from components.dashboard import render_dashboard
from components.stock_analysis import render_stock_analysis
from services.gemini_service import run_financial_agent
from components.landing_page import render_landing_page

# First Streamlit command must be set_page_config
st.set_page_config(
    page_title=f"FinquoraAI | Advanced Financial Analysis & Prediction Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://github.com/abrarwarunkar/Finquora.ai',
        'Report a bug': "https://github.com/abrarwarunkar/Finquora.ai/issues",
        'About': """
        # FinquoraAI
        Smart Financial Insights for Every Step
        
        An advanced financial analysis and prediction platform powered by artificial intelligence.
        """
    }
)

# Import after environment variables are set
# Update this line in your app.py

# ==============================================
# BRANDING CONFIGURATION
# ==============================================
# Update BRAND_CONFIG
# In BRAND_CONFIG, update the logo path
import os

# At the top of your file, add:
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(CURRENT_DIR, "static", "images", "finquora_logo.png")

BRAND_CONFIG = {
    "name": "FinquoraAI",
    "tagline": "Advanced Financial Analysis & Prediction Platform",
    "logo": LOGO_PATH,  # Using the constructed path
    "colors": {
        "primary": "#2962ff",    # Vibrant blue
        "secondary": "#757575",   # Neutral gray
        "accent": "#00e676",     # Fresh green
        "dark": "#1a1a1a",       # Deep dark
        "light": "#ffffff",      # Pure white
        "success": "#00c853",    # Rich green
        "warning": "#ffd600",    # Bright yellow
        "danger": "#ff1744"      # Vivid red
    },
    "font": {
        "main": "'Poppins', 'Helvetica Neue', Arial, sans-serif",
        "mono": "'JetBrains Mono', monospace"
    }
}

# ==============================================
# PAGE CONFIGURATION
# ==============================================
st.set_page_config(
    page_title=f"{BRAND_CONFIG['name']} | {BRAND_CONFIG['tagline']}",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=BRAND_CONFIG['logo'],
    menu_items={
        'Get Help': 'https://github.com/abrarwarunkar/Finquora.ai',
        'Report a bug': "https://github.com/abrarwarunkar/Finquora.ai/issues",
        'About': """
        # FinquoraAI
        Smart Financial Insights for Every Step
        
        An advanced financial analysis and prediction platform powered by artificial intelligence.
        """
    }
)

# Add OpenGraph metadata right after the page config
st.markdown("""
    <head>
        <title>FinquoraAI - Smart Financial Insights</title>
        <meta name="description" content="Advanced Financial Analysis & Prediction Platform" />
        <meta property="og:title" content="FinquoraAI" />
        <meta property="og:description" content="Advanced Financial Analysis & Prediction Platform" />
        <meta property="og:url" content="https://finquora-ai.streamlit.app/" />
    </head>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = set()

if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = {}

if 'messages' not in st.session_state:
    st.session_state.messages = []

def main():
    # Update the brand header with st.image
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(BRAND_CONFIG['logo'], width=100)
    with col2:
        st.markdown(f"""
        <h1 class="brand-name">{BRAND_CONFIG['name']}</h1>
        <p class="brand-tagline">{BRAND_CONFIG['tagline']}</p>
        """, unsafe_allow_html=True)

    # Update sidebar brand with st.image
    st.sidebar.image(BRAND_CONFIG['logo'], width=50)
    st.sidebar.markdown(f"""
    <div class="sidebar-brand-text">
        <div class="sidebar-brand-name">{BRAND_CONFIG['name']}</div>
        <div class="sidebar-brand-tagline">Market Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    # Ticker Input
    ticker = st.sidebar.text_input(
        "Enter stock or crypto symbol", 
        key="ticker_input",
        placeholder="e.g., AAPL, MSFT, BTC-USD",
        help="Enter a valid stock ticker or cryptocurrency symbol"
    )
    
    # Market Overview
    st.sidebar.subheader("🌎 Market Overview")
    from services.stock_service import get_market_indices
    get_market_indices()
    
    # Portfolio Management Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Portfolio Management")
    if ticker:
        if st.sidebar.button("Add to Portfolio", key="add_portfolio_btn"):
            quantity = st.sidebar.number_input("Quantity", min_value=1, value=1, key="qty_input")
            entry_price = st.sidebar.number_input("Entry Price ($)", min_value=0.01, value=100.00, key="price_input")
            st.session_state.portfolio[ticker] = {
                "quantity": quantity,
                "entry_price": entry_price
            }
            st.sidebar.success(f"Added {ticker} to portfolio!")

    # Display Portfolio Summary
    if st.session_state.portfolio:
        for symbol, details in st.session_state.portfolio.items():
            st.sidebar.markdown(f"""
            **{symbol}**
            - Quantity: {details['quantity']}
            - Entry: ${details['entry_price']:.2f}
            """)
            if st.sidebar.button(f"Remove", key=f"remove_portfolio_{symbol}"):
                del st.session_state.portfolio[symbol]
                st.rerun()

    # Watchlist Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("👀 Watchlist")
    if ticker:
        if st.sidebar.button("Add to Watchlist", key="add_watchlist_btn"):
            st.session_state.watchlist.add(ticker)
            st.sidebar.success(f"Added {ticker} to watchlist!")

    # Display Watchlist
    if st.session_state.watchlist:
        for symbol in st.session_state.watchlist:
            col1, col2 = st.sidebar.columns([3,1])
            with col1:
                st.markdown(f"• {symbol}")
            with col2:
                if st.sidebar.button("×", key=f"remove_watch_{symbol}"):
                    st.session_state.watchlist.remove(symbol)
                    st.rerun()

    # Price Alerts Section
    with st.sidebar.expander("🔔 Price Alerts", expanded=True):
        if ticker:
            alert_price = st.number_input("Alert Price ($)", min_value=0.01, value=100.00, key="alert_price_input")
            alert_type = st.selectbox("Alert Type", ["Above", "Below"], key="alert_type_select")
            
            if st.button("Set Alert", key="set_alert_btn"):
                st.session_state.price_alerts[ticker] = {
                    "price": alert_price,
                    "type": alert_type
                }
                st.success(f"Alert set for {ticker}!")
        
        if st.session_state.price_alerts:
            st.markdown("### Active Alerts")
            for symbol, alert in st.session_state.price_alerts.items():
                st.markdown(f"""
                **{symbol}** - {alert['type']} ${alert['price']:.2f}
                """)
                if st.button("Remove Alert", key=f"remove_alert_{symbol}"):
                    del st.session_state.price_alerts[symbol]
                    st.rerun()

    # ==============================================
    # MAIN CONTENT AREA
    # ==============================================
    if not ticker:
        render_landing_page()
    else:
        try:
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Market Snapshot", 
                "📅 Corporate Calendar", 
                "📈 Technical Analysis",
                "🔮 Stock Prediction"
            ])
            
            with tab1:
                render_dashboard(ticker)
            
            with tab2:
                from components.stock_analysis import render_corporate_calendar
                render_corporate_calendar(ticker)
            
            with tab3:
                render_stock_analysis(ticker)
            
            with tab4:
                from components.stock_analysis import render_stock_prediction
                render_stock_prediction(ticker)
                
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")

    # ==============================================
    # CHAT INTERFACE (Single Instance)
    # ==============================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 Financial Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with unique key
    if prompt := st.chat_input(
        "Ask about financial markets, analysis, or investment strategies...",
        key="financial_chat_input"
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = run_financial_agent(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    from streamlit.components.v1 import html
    from utils.visualization import get_fullscreen_js
    html(get_fullscreen_js())
    main()


    
   