import os
import torch
import torch.nn as nn
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv
from model import SP500LSTM
load_dotenv()

TICKER = 'SPY'
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
MODEL_PATH = os.path.join('models', f'{TICKER}.pth')
SCALER_PATH = os.path.join('scalers', f'{TICKER}.npy')

SEQUENCE_LENGTH = 20
FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'Return', 'RSI', 'MACD']

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

def download_data(ticker):
    df = yf.download(ticker, period='40d', interval='1d')
    df['Return'] = df['Close'].pct_change()
    df = compute_technical_indicators(df)
    df.dropna(inplace=True)
    return df

def load_scaler(path):
    scaler_params = np.load(path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']
    return scaler

def prepare_input(df, scaler):
    features = df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)

    x_input = features_scaled[-SEQUENCE_LENGTH:]
    x_input = np.expand_dims(x_input, axis=0)
    x_tensor = torch.tensor(x_input, dtype=torch.float32)
    return x_tensor

def load_model(path):
    model = SP500LSTM(input_size=len(FEATURE_COLUMNS), hidden_size=64, num_layers=2, output_size=2)
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def send_discord_message(direction, confidence):
    if not DISCORD_WEBHOOK_URL:
        print('Discord webhook URL not set')
        return
    emoji = "📈" if direction == 1 else "📉"
    message = f'Daily {TICKER} Signal\nDirection: {emoji} {"Up" if direction == 1 else "Down"}\nConfidence: {confidence*100:.2f}%'
    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content=message)
    webhook.execute()

def main():
    df = download_data(TICKER)
    scaler = load_scaler(SCALER_PATH)
    x = prepare_input(df, scaler)
    model = load_model(MODEL_PATH)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        direction = int(np.argmax(probs))
        confidence = probs[direction]

    print(f'Predicted direction: {direction}, confidence: {confidence:.2%}')
    send_discord_message(direction, confidence)

if __name__ == "__main__":
    main()
