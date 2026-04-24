# Finquora.ai 📈

![Finquora AI](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E)
![Quantitative Finance](https://img.shields.io/badge/Quant-SciPy-0054A6)
![AI](https://img.shields.io/badge/AI-Groq_LLaMA3-black)

**Finquora.ai** is an institutional-grade financial analysis dashboard powered by Artificial Intelligence, Machine Learning, and advanced Quantitative Mathematics. Designed to move beyond basic retail trading indicators, Finquora provides users with sophisticated tools for risk management, price prediction, and real-time market sentiment synthesis.

---

## 🚀 Key Features

### 🧮 Quantitative Finance (NSE-Tier)
* **Black-Scholes Options Pricing:** Calculates theoretical European option prices (Calls/Puts) based on Spot, Strike, Volatility, Time to Expiry, and Risk-Free Rate.
* **The Greeks:** Mathematically derives and displays sensitivity metrics (Delta, Gamma, Theta, Vega, Rho) alongside interactive Delta-Spot curve visualizations.
* **Modern Portfolio Theory (Markowitz):** A Portfolio Risk Optimizer that runs Monte Carlo simulations to plot the Efficient Frontier. It utilizes the SLSQP optimization algorithm to automatically calculate the exact asset allocation weights required to maximize the **Sharpe Ratio**.

### 🔮 Machine Learning Models
* **Anomaly Detection (Isolation Forests):** An unsupervised machine learning pipeline that trains on historical price and volume data to automatically flag and plot highly unusual market behavior.
* **Price Prediction (Random Forests):** A supervised regression model trained on technical indicators (Moving Averages, RSI) to forecast short-term future price action with visualized confidence intervals and accuracy scoring (RMSE, MAE).

### 🧠 Artificial Intelligence (Powered by Groq)
* **Real-time News Sentiment Analysis:** Fetches breaking news headlines via `yfinance` and feeds them into the lightning-fast **LLaMA 3.1 8B** model to instantly synthesize market mood and generate Bullish/Bearish confidence scores.
* **Financial Chat Assistant:** A persistent conversational agent powered by the massive **LLaMA 3.3 70B** model, acting as an expert financial advisor for deep market inquiries.

---

## 🛠️ Technology Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (with custom CSS Glassmorphism & premium UI/UX)
* **Data Visualization:** [Plotly](https://plotly.com/python/) (Interactive Financial Charts)
* **Market Data:** [yfinance](https://pypi.org/project/yfinance/) (Yahoo Finance API)
* **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (RandomForestRegressor, IsolationForest)
* **Quantitative Math:** [SciPy](https://scipy.org/), [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
* **Large Language Models:** [Groq API](https://groq.com/) (LLaMA 3 Architecture)

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abrarwarunkar/Finquora.ai.git
   cd Finquora.ai
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the `streamlit_app` directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the Application:**
   ```bash
   python -m streamlit run streamlit_app/app.py
   ```

---

## 🧭 Navigation & Usage

1. **The Landing Page:** You will be greeted by a central, intuitive search bar. Simply type in any valid stock or cryptocurrency ticker (e.g., `AAPL`, `NVDA`, `BTC-USD`) and press Enter.
2. **The Dashboard:** The app will seamlessly transition to the main dashboard, revealing 5 advanced analysis tabs:
   - ML Price Prediction
   - AI Sentiment Analysis
   - ML Anomaly Detection
   - Options Pricing
   - Portfolio Optimization
3. **The AI Assistant:** At any time, you can open the sidebar to chat directly with the Groq-powered financial assistant to ask questions about your findings.

---
*Disclaimer: Finquora.ai is a technical demonstration of quantitative and machine learning concepts. The predictions and analyses provided by the software are for educational purposes only and do not constitute financial advice.*
