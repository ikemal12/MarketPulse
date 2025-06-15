import os
import pandas as pd
import numpy as np
import torch
import argparse
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

SEQUENCE_LENGTH = 20

def compute_technical_indicators(df):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df

def prepare_dataset(ticker):
    print(f"Preparing data for: {ticker}")
    os.makedirs('datasets', exist_ok=True)
    os.makedirs('scalers', exist_ok=True)

    df = yf.download(ticker, period='5y', interval='1d')
    df = df.dropna()
    df = compute_technical_indicators(df)
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)
    df = df.dropna()

    FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'Return', 'RSI', 'MACD']
    scaler = StandardScaler()
    features = scaler.fit_transform(df[FEATURE_COLUMNS])
    scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
    np.save(f'scalers/scaler_{ticker}.npy', scaler_params)

    x, y = [], []
    for i in range(SEQUENCE_LENGTH, len(features)):
        x.append(features[i - SEQUENCE_LENGTH:i])
        y.append(df['Target'].iloc[i])

    x = np.array(x)
    y = np.array(y)

    print(f'Sequences created: {x.shape}, Labels: {y.shape}')
    unique, counts = np.unique(y, return_counts=True)
    print('Label distribution:', dict(zip(unique, counts)))

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    torch.save((X_train, y_train, X_test, y_test), f'datasets/dataset_{ticker}.pt')
    print(f'Dataset saved to datasets/dataset_{ticker}.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker symbol (default: SPY)')
    args = parser.parse_args()
    prepare_dataset(args.ticker.upper())
