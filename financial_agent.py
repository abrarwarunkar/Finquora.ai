import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

# Get the API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

#web search-agent
web_search_agent=Agent(
    name="web_search_Agent",
    role="Search the web for information",
    model=Groq(id="llama3-70b-8192"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources."],
    show_tool_calls=True,
    markdown=True,
)

#Financial agent
financial_agent=Agent(
    name="Finance AI Agent",
    role="Analyze financial data and provide insights",
    model=Groq(id="llama3-70b-8192"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True),
        ],
    instructions=["Use tables to display data."],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent=Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama3-70b-8192"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)
multi_ai_agent.print_response("Summarize analyst recommendations and share the latest news for NVDA", stream=True)