import requests
import json
import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

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
            # Fix futurewarning by using .iloc[-1, 0] if spy['Close'] is a DataFrame in newer yfinance, 
            # or just simple selection for older ones.
            close_col = spy['Close']
            if isinstance(close_col, pd.DataFrame):
                val = close_col.iloc[:, 0]
            else:
                val = close_col
            pct_change = val.pct_change().iloc[-1]
            if isinstance(pct_change, pd.Series):
                pct_change = pct_change.iloc[0]
            context.append(f"Recent S&P 500 movement: {pct_change:.2%}")
    except Exception as e:
        print(f"Market data error: {e}")

    return "\n".join(context)

def run_financial_agent(prompt):
    """Enhanced financial analysis with RAG architecture using Groq API"""
    try:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY is not set in environment variables."

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

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional financial advisor and market analyst."
                },
                {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "I apologize, but I couldn't generate a response for that query."
            
    except requests.exceptions.HTTPError as e:
        return f"API HTTP Error: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error processing request: {str(e)}"

def analyze_news_sentiment(news_items, ticker):
    """Analyze the sentiment of a list of news items using Groq API"""
    try:
        if not GROQ_API_KEY:
            return "Error: GROQ_API_KEY is not set."
            
        if not news_items:
            return "No news available to analyze."
            
        formatted_news = []
        for item in news_items[:10]:
            content = item.get('content')
            if content and isinstance(content, dict):
                title = content.get('title') or 'No Title'
                summary = content.get('summary') or 'No Summary'
            else:
                title = item.get('title') or 'No Title'
                summary = item.get('summary') or 'No Summary'
            formatted_news.append(f"- {title}: {summary}")
            
        news_text = "\n".join(formatted_news)
        
        prompt = f"""
You are an expert financial AI. Analyze the following recent news headlines and summaries for {ticker}.
Based ONLY on these news items, determine the current market sentiment for {ticker}.

News Items:
{news_text}

Provide:
1. Overall Sentiment (Bullish, Bearish, or Neutral) and a confidence score out of 100%.
2. A 3-4 sentence summary explaining the key drivers of this sentiment.
3. Any potential risks mentioned in the news.

Keep the response concise and formatted using Markdown.
"""

        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "llama-3.1-8b-instant", # Using fast 8b model for sentiment
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 512
        }
        
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            return "Could not generate sentiment analysis."
            
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"
