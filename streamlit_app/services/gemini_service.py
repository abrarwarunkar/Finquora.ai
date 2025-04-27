import requests
import json
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_financial_context(query):
    """Multi-source knowledge querying"""
    context = []
    
    # News Source Integration
    try:
        # You would need to add news API keys and implementation
        context.append("News data from financial sources")
    except Exception as e:
        print(f"News retrieval error: {e}")

    # Market Data
    try:
        # Basic market data from yfinance
        if any(symbol in query.upper() for symbol in ['STOCK', 'TICKER', 'PRICE']):
            spy = yf.download('^GSPC', start=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'))
            context.append(f"Recent S&P 500 movement: {spy['Close'].pct_change().iloc[-1]:.2%}")
    except Exception as e:
        print(f"Market data error: {e}")

    return "\n".join(context)

def run_financial_agent(prompt):
    """Enhanced financial analysis with RAG architecture"""
    try:
        # Get relevant financial context
        context = get_financial_context(prompt)
        
        # Construct enhanced prompt
        enhanced_prompt = f"""Context: {context}

Query: {prompt}

Please provide a comprehensive financial analysis considering:
1. Market context and trends
2. Technical indicators if relevant
3. News sentiment
4. Potential risks and opportunities
5. Clear recommendations or insights

Analysis:"""

        url = f"{BASE_URL}?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": enhanced_prompt
                }]
            }]
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "I apologize, but I couldn't generate a response for that query."
            
    except Exception as e:
        return f"Error processing request: {str(e)}"