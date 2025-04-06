import streamlit as st
from financial_agent import financial_agent
from dotenv import load_dotenv
import os
import yfinance as yf
import json
import re

# Load .env
load_dotenv()

# Page settings
st.set_page_config(page_title="Finance AI Chatbot 💬", layout="centered")

# Custom dark style with white search bar
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: #f1f1f1;
    }
    .css-1d391kg, .css-18ni7ap, .st-bb, .st-at {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
    }
    /* Style for the chat input box */
    .stChatInputContainer {
        background-color: white !important;
        border-radius: 10px;
    }
    .stChatInput {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #cccccc;
    }
    .stChatInput::placeholder {
        color: #666666 !important;
    }
    /* Style for the sidebar input */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #333333 !important;
        border: 1px solid #cccccc;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("💸 Finance AI Chatbot")
st.caption("Powered by PhiData AI Agent | Ask about stocks, crypto, personal finance & more!")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Helper function to process agent response
def process_agent_response(response):
    """Extract the clean final response and remove 'Running:' lines and tool calls"""
    # Convert response to string if it's not already
    if not isinstance(response, str):
        if hasattr(response, 'content') and response.content:
            response = response.content
        elif hasattr(response, 'messages'):
            # Try to get last assistant message
            for msg in reversed(response.messages):
                if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content') and msg.content:
                    response = msg.content
                    break
            else:
                response = str(response)
        else:
            response = str(response)
    
    # Process the string response
    
    # 1. Check if it starts with "Running:" and has tool calls
    if "Running:" in response:
        # Split by "Running:" to remove the header
        parts = response.split("Running:", 1)
        if len(parts) > 1:
            response = parts[1]
        
        # Remove all tool call lines (lines starting with *)
        lines = response.split('\n')
        clean_lines = []
        skip_tool_calls = True
        
        for line in lines:
            stripped = line.strip()
            # Skip lines starting with * which indicate tool calls
            if stripped.startswith('*'):
                continue
            # Once we find a non-tool call line, start including content
            if stripped and not stripped.startswith('*'):
                skip_tool_calls = False
            if not skip_tool_calls:
                clean_lines.append(line)
        
        response = '\n'.join(clean_lines).strip()
    
    # 2. Handle content= format that may appear
    if "content=\"" in response:
        content_match = re.search(r'content="(.*?)(?:" content_type|$)', response, re.DOTALL)
        if content_match:
            response = content_match.group(1).replace('\\n', '\n').replace('\\"', '"')
    
    # Final cleanup
    return response.strip()

# Chat input
if user_input := st.chat_input("Ask me a finance question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get and show agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            raw_response = financial_agent.run(user_input)
            clean_response = process_agent_response(raw_response)
            st.markdown(clean_response)

    # Save clean response
    st.session_state.messages.append({"role": "assistant", "content": clean_response})

# Sidebar stock chart
st.sidebar.title("📊 Stock Snapshot")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., AAPL, TSLA)")

if ticker:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        st.sidebar.line_chart(hist["Close"], use_container_width=True)
        st.sidebar.write(stock.info.get("longName", ""))
        st.sidebar.metric("Current Price", f"${hist['Close'][-1]:.2f}")
    except Exception as e:
        st.sidebar.error("Could not fetch data for that ticker.")