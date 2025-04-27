import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add this function or modify your existing apply_custom_css function

# Remove this entire function
def get_chat_js():
    """Get JavaScript for chat popup functionality"""
    return """
    <script>
        // Wait for the DOM to be fully loaded
        window.addEventListener('load', function() {
            // Get elements
            const chatToggle = document.querySelector('.chat-toggle-btn');
            const chatPopup = document.querySelector('.chat-popup');
            const closeChat = document.querySelector('.close-chat-btn');
            
            if (chatToggle && chatPopup) {
                // Toggle chat function
                function toggleChat() {
                    chatPopup.classList.toggle('active');
                    chatToggle.classList.toggle('active');
                }
                
                // Add click event listeners
                chatToggle.addEventListener('click', toggleChat);
                if (closeChat) {
                    closeChat.addEventListener('click', toggleChat);
                }
            }
        });
    </script>
    """

# Modify the apply_custom_css function to include the JavaScript
def apply_custom_css(BRAND_CONFIG):
    custom_css = f"""
    <style>
        /* Logo styling */
        .brand-logo-container {{
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .brand-logo {{
            width: 120px;
            height: auto;
            margin-bottom: 1rem;
        }}
        
        .sidebar-logo {{
            width: 40px;
            height: auto;
            margin-right: 1rem;
        }}
        
        /* Brand header styling */
        .brand-header {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        /* Sidebar brand styling */
        .sidebar-brand {{
            display: flex;
            align-items: center;
            padding: 1rem;
            margin-bottom: 2rem;
            background: linear-gradient(120deg, {BRAND_CONFIG['colors']['primary']}22, {BRAND_CONFIG['colors']['accent']}22);
            border-radius: 0.5rem;
        }}
        
        .sidebar-brand-text {{
            flex: 1;
        }}
        
        /* Existing styles... */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(get_chat_js(), unsafe_allow_html=True)

def get_fullscreen_js():
    """Get JavaScript for fullscreen functionality"""
    return """
    <script>
    function toggleFullscreen() {
        const sidebarContent = window.parent.document.querySelector('.sidebar .sidebar-content');
        sidebarContent.classList.toggle('fullscreen-mode');
        
        const icon = document.getElementById('fullscreen-icon');
        if (sidebarContent.classList.contains('fullscreen-mode')) {
            icon.innerHTML = '🗗 Exit Fullscreen';
            icon.style.backgroundColor = '#dc3545';
            const overlay = document.createElement('div');
            overlay.id = 'fullscreen-overlay';
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100%';
            overlay.style.height = '100%';
            overlay.style.background = 'rgba(0,0,0,0.7)';
            overlay.style.zIndex = '9998';
            overlay.onclick = toggleFullscreen;
            document.body.appendChild(overlay);
        } else {
            icon.innerHTML = '🗖 Fullscreen';
            icon.style.backgroundColor = '';
            const overlay = document.getElementById('fullscreen-overlay');
            if (overlay) overlay.remove();
        }
        window.dispatchEvent(new Event('resize'));
    }

    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            const sidebarContent = window.parent.document.querySelector('.sidebar .sidebar-content.fullscreen-mode');
            if (sidebarContent) toggleFullscreen();
        }
    });
    </script>
    """

def create_stock_chart(data):
    """Create an interactive stock chart using Plotly"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Add moving averages if available
    if 'MA' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MA'],
                name='Moving Average',
                line=dict(color='rgba(255, 207, 102, 1)', width=2)
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title='Stock Price Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_portfolio_chart(portfolio_data):
    # Extract data
    symbols = [item['Symbol'] for item in portfolio_data]
    values = [float(item['Value'].replace('$', '').replace(',', '')) for item in portfolio_data]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        hole=.4,
        textinfo='label+percent',
        marker=dict(
            colors=['#2962ff', '#00e676', '#ff6d00', '#aa00ff', '#2979ff', '#00e5ff', '#76ff03', '#ffea00', '#ff4081'],
            line=dict(color='#ffffff', width=2)
        )
    )])
    
    fig.update_layout(
        title='Portfolio Allocation',
        template='plotly_dark',
        height=400,
        annotations=[dict(text='Portfolio', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def create_market_overview_chart(market_data):
    fig = go.Figure()
    
    for idx, data in market_data.items():
        # Normalize to percentage change from first day
        normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
        
        # Map index symbol to readable name
        index_names = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^RUT": "Russell 2000"
        }
        name = index_names.get(idx, idx)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=normalized,
            mode='lines',
            name=name
        ))
    
    fig.update_layout(
        title='Market Indices Performance (% Change)',
        xaxis_title='Date',
        yaxis_title='% Change',
        template='plotly_dark',
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig