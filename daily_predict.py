import os
import torch
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from discord import Webhook, RequestsWebhookAdapter

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
MODEL_PATH = 'sp500_lstm.pth'

SEQUENCE_LENGTH = 20
FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'Return']

def download_data():
    ticker = '^GSPC'
    df = yf.download(ticker, period='40d', interval='1d')
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df

def prepare_input(df, scaler):
    features = df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)

    x_input = features_scaled[-SEQUENCE_LENGTH:]
    x_input = np.expand_dims(x_input, axis=0)  
    x_tensor = torch.tensor(x_input, dtype=torch.float32)
    return x_tensor

def load_scaler(path='scaler.npy'):
    scaler_params = np.load(path, allow_pickle=True).item()
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean']
    scaler.scale_ = scaler_params['scale']
    return scaler

def load_model(path=MODEL_PATH):
    model = torch.load(path)
    model.eval()
    return model

def send_discord_message(direction, confidence):
    if not DISCORD_WEBHOOK_URL:
        print("Discord webhook URL not set")
        return
    webhook = Webhook.from_url(DISCORD_WEBHOOK_URL, adapter=RequestsWebhookAdapter())
    emoji = "📈" if direction == "Up" else "📉"
    message = f'Daily S&P 500 Signal\nDirection: {emoji} {'Up' if direction == 1 else 'Down'}\nConfidence: {confidence*100:.2f}%'
    webhook.send(message)

def main():
    df = download_data()
    scaler = load_scaler()
    x = prepare_input(df, scaler)
    model = load_model()

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        direction = int(np.argmax(probs))
        confidence = probs[direction]

    print(f'Predicted direction: {direction}, confidence: {confidence}')
    send_discord_message(direction, confidence)

if __name__ == "__main__":
    main()