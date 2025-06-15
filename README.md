# MarketPulse

MarketPulse is a Discord bot that provides real-time stock market signals, news, statistics, and visualizations based on LSTM deep learning models. It helps traders and enthusiasts stay informed with up-to-date predictions and market insights directly within Discord.

## Features

- **Signal Predictions:** Get buy/sell signals with confidence scores for popular stocks and funds.  
- **Financial News:** Fetches the latest relevant news articles for specified tickers.  
- **Market Statistics:** Displays key metrics such as daily returns, volatility, and moving averages.  
- **Price Charts:** Generates candlestick charts with moving averages for visual trend analysis.  
- **Multi-Ticker Support:** Easily query predictions and info for multiple tickers via slash commands.

## Currently Supported Tickers (examples)

- SPY (S&P 500 ETF)  
- AAPL (Apple Inc.)  
- TSLA (Tesla Inc.)  
- GOLD (Gold Spot Price)  
- SILVER (Silver Spot Price)  
- OIL (Crude Oil Price)

## Tech Stack

- **PyTorch** for LSTM model training and inference  
- **Discord.py** for bot integration and slash commands  
- **NewsAPI** for fetching financial news  
- **Matplotlib & mplfinance** for generating financial charts  
- **yFinance** for data download and preprocessing  

---

## Getting Started

1. Clone the repository  
2. Create and activate a Python virtual environment  
3. Install dependencies via `pip install -r requirements.txt`  
4. Set up your `.env` file with your Discord bot token, NewsAPI key, and guild ID  
5. Train models for desired tickers or use pretrained models in `models/` and `datasets/`  
6. Run `bot.py` to start the MarketPulse Discord bot  

---

Feel free to contribute or open issues for new features or bugs!


