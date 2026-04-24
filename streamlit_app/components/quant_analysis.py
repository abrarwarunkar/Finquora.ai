import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.quant_utils import black_scholes_calc, optimize_portfolio

def render_options_pricing(ticker):
    st.subheader(f"🧮 Options Pricing & Greeks (Black-Scholes)")
    st.markdown("Calculate theoretical option prices and sensitivities (Greeks) using the Black-Scholes model.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Option Parameters**")
        # Default spot to current price if possible
        import yfinance as yf
        try:
            current_price = yf.Ticker(ticker).fast_info.last_price
        except:
            current_price = 100.0
            
        S = st.number_input("Spot Price ($)", min_value=0.01, value=float(current_price), step=1.0)
        K = st.number_input("Strike Price ($)", min_value=0.01, value=float(current_price), step=1.0)
        T_days = st.number_input("Days to Expiry", min_value=1, value=30, step=1)
        r_pct = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=2.0, step=0.1)
        sigma_pct = st.number_input("Volatility (%)", min_value=0.1, value=20.0, step=1.0)
        option_type = st.selectbox("Option Type", ["Call", "Put"]).lower()
        
        T = T_days / 365.0
        r = r_pct / 100.0
        sigma = sigma_pct / 100.0
        
    with col2:
        if st.button("Calculate Theoretical Price"):
            results = black_scholes_calc(S, K, T, r, sigma, option_type)
            if results:
                st.markdown("### Results")
                st.metric("Theoretical Price", f"${results['Price']:.2f}")
                
                # Display Greeks in a grid
                st.markdown("#### The Greeks")
                g1, g2, g3 = st.columns(3)
                g1.metric("Delta (Δ)", f"{results['Delta']:.4f}")
                g2.metric("Gamma (Γ)", f"{results['Gamma']:.4f}")
                g3.metric("Theta (Θ)", f"{results['Theta']:.4f}")
                
                g4, g5, g6 = st.columns(3)
                g4.metric("Vega (ν)", f"{results['Vega']:.4f}")
                g5.metric("Rho (ρ)", f"{results['Rho']:.4f}")
                
                # Plot Delta curve
                spots = np.linspace(S * 0.7, S * 1.3, 100)
                deltas = []
                for spot in spots:
                    res = black_scholes_calc(spot, K, T, r, sigma, option_type)
                    deltas.append(res['Delta'] if res else 0)
                    
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=spots, y=deltas, name='Delta Curve', line=dict(color='#2962ff')))
                fig.add_vline(x=S, line_dash="dash", line_color="red", annotation_text="Current Spot")
                fig.update_layout(title="Delta vs Spot Price", xaxis_title="Spot Price ($)", yaxis_title="Delta", height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

def render_portfolio_optimization():
    st.subheader("⚖️ Modern Portfolio Theory (Efficient Frontier)")
    st.markdown("Optimize a portfolio of stocks to maximize the Sharpe Ratio (risk-adjusted return) using the Markowitz Efficient Frontier.")
    
    tickers_input = st.text_input("Enter comma-separated tickers", "AAPL, MSFT, GOOGL, NVDA")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers.")
        return
        
    if st.button("Optimize Portfolio"):
        with st.spinner("Calculating covariance matrix and running simulation (10,000 portfolios)..."):
            results = optimize_portfolio(tickers)
            if not results:
                st.error("Could not fetch sufficient data for the provided tickers.")
                return
                
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Optimal Allocation")
                st.markdown("*Maximum Sharpe Ratio Portfolio*")
                
                # Pie chart of weights
                labels = list(results['optimal_weights'].keys())
                values = list(results['optimal_weights'].values())
                
                fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
                fig_pie.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.metric("Expected Annual Return", f"{results['optimal_return']*100:.2f}%")
                st.metric("Annual Volatility (Risk)", f"{results['optimal_volatility']*100:.2f}%")
                st.metric("Sharpe Ratio", f"{results['optimal_sharpe']:.2f}")
                
            with col2:
                # Efficient Frontier Plot
                fig = go.Figure()
                
                # Random portfolios
                fig.add_trace(go.Scatter(
                    x=results['frontier_volatility'], 
                    y=results['frontier_returns'], 
                    mode='markers',
                    marker=dict(
                        color=results['frontier_sharpe'], 
                        colorscale='Viridis', 
                        showscale=True,
                        size=5,
                        colorbar=dict(title='Sharpe Ratio')
                    ),
                    name='Simulated Portfolios'
                ))
                
                # Optimal portfolio marker
                fig.add_trace(go.Scatter(
                    x=[results['optimal_volatility']],
                    y=[results['optimal_return']],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Max Sharpe Ratio'
                ))
                
                fig.update_layout(
                    title="Markowitz Efficient Frontier",
                    xaxis_title="Volatility (Risk)",
                    yaxis_title="Expected Return",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
