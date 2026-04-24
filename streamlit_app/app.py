# First, import only non-streamlit modules
import os
from dotenv import load_dotenv

# Set up configuration before any Streamlit imports
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

# Now import streamlit and set page config
import streamlit as st

# First Streamlit command
st.set_page_config(
    page_title=f"{BRAND_CONFIG['name']} | {BRAND_CONFIG['tagline']}",
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

# Import other components after page config
from components.dashboard import render_dashboard

from services.groq_service import run_financial_agent
from components.landing_page import render_landing_page

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
        "🔍 Search Ticker (e.g. AAPL)", 
        key="ticker_input",
        placeholder="AAPL, NVDA, BTC-USD...",
        help="Enter a valid stock ticker or cryptocurrency symbol"
    )
    # ==============================================
    # MAIN CONTENT AREA
    # ==============================================
    if not ticker:
        render_landing_page()
    else:
        try:
            # Create AI/ML focused tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔮 ML Price Prediction", 
                "🧠 AI Sentiment Analysis", 
                "🔍 ML Anomaly Detection",
                "🧮 Options Pricing",
                "⚖️ Portfolio Optimization"
            ])
            
            with tab1:
                from components.stock_analysis import render_stock_prediction
                render_stock_prediction(ticker)
            
            with tab2:
                from components.stock_analysis import render_ai_sentiment_analysis
                render_ai_sentiment_analysis(ticker)
            
            with tab3:
                from components.stock_analysis import render_ml_anomaly_detection
                render_ml_anomaly_detection(ticker)
                
            with tab4:
                from components.quant_analysis import render_options_pricing
                render_options_pricing(ticker)
                
            with tab5:
                from components.quant_analysis import render_portfolio_optimization
                render_portfolio_optimization()
                
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

main()