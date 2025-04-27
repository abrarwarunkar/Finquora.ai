import streamlit as st

def render_landing_page():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style='font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem;'>
            Welcome to QuantivueAI
        </h1>
        <p style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>
            Your Advanced Financial Analysis & Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Market Analysis
        - Real-time market data
        - Technical indicators
        - Trend analysis
        - Volume analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI-Powered Insights
        - Smart predictions
        - Sentiment analysis
        - Risk assessment
        - Market intelligence
        """)
    
    with col3:
        st.markdown("""
        ### 💼 Portfolio Management
        - Track investments
        - Set price alerts
        - Watch favorite stocks
        - Performance metrics
        """)
    
    # Getting Started Section
    st.markdown("---")
    st.markdown("""
    ## 🚀 Getting Started
    Enter a stock symbol in the sidebar to begin your analysis journey.
    """)
    
    # Key Benefits
    st.markdown("---")
    st.markdown("## Why Choose QuantivueAI?")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric("Real-time Data", "✓")
    with metrics_col2:
        st.metric("AI Analysis", "✓")
    with metrics_col3:
        st.metric("Portfolio Tracking", "✓")
    with metrics_col4:
        st.metric("Smart Alerts", "✓")