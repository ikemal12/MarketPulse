import yfinance as yf
import pandas as pd

# download data
ticker = '^GSPC'  
df = yf.download(ticker, start='2020-01-01', end=None)
df = df.dropna()

# add daily returns
df['Return'] = df['Close'].pct_change()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df[:-1]

# save to csv
df.to_csv('sp500.csv')
print("Data saved")