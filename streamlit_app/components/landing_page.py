import streamlit as st

def render_landing_page():
    # Premium Hero Section with Glassmorphism / Gradients
    st.markdown("""
    <style>
    .hero-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(41, 98, 255, 0.05) 0%, rgba(0, 230, 118, 0.05) 100%);
        border-radius: 20px;
        margin-bottom: 3rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #2962ff, #00e676);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0a0;
        margin-bottom: 2rem;
        font-weight: 400;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    
    <div class="hero-container">
        <h1 class="hero-title">Institutional Intelligence.</h1>
        <h1 class="hero-title" style="margin-top:-10px; font-size:2.8rem;">Now Accessible.</h1>
        <p class="hero-subtitle">
            Advanced Machine Learning, Quantitative Mathematics, and AI-Driven Financial Analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Central Search Bar
    st.markdown("<h4 style='text-align: center; margin-bottom: 1rem;'>Enter a Ticker to Begin Analysis</h4>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        def update_sidebar_ticker():
            st.session_state.ticker_input = st.session_state.center_search
            
        st.text_input(
            "Search Ticker",
            placeholder="e.g., AAPL, NVDA, BTC-USD...",
            key="center_search",
            label_visibility="collapsed",
            on_change=update_sidebar_ticker
        )
        st.markdown("<p style='text-align: center; color: #666; font-size: 0.85rem;'>Press Enter to launch dashboard</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem; margin-top: 1rem;'>Platform Capabilities</h2>", unsafe_allow_html=True)
    
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("""
        ### 🧠 Groq AI Engine
        Powered by the lightning-fast LLaMA 3.3 architecture.
        - **Real-time News Sentiment:** Instant synthesis of market headlines.
        - **Financial Chat Assistant:** Conversational access to deep financial concepts.
        """)
    
    with col_feat2:
        st.markdown("""
        ### 🔮 Machine Learning
        Supervised and unsupervised models for predictive insights.
        - **Random Forest Prediction:** Forecasting short-term price movements.
        - **Isolation Forest:** Automated detection of anomalous trading volume and price action.
        """)
    
    with col_feat3:
        st.markdown("""
        ### 🧮 Quantitative Finance
        Institutional-grade mathematical modeling and risk management.
        - **Options Pricing:** Black-Scholes theoretical pricing and dynamic Greeks.
        - **Modern Portfolio Theory:** Markowitz Efficient Frontier optimization (Max Sharpe).
        """)