import os
import torch
import torch.nn.functional as F
import discord
import humanize
from discord import app_commands
from dotenv import load_dotenv
from newsapi import NewsApiClient
from scripts.model import SP500LSTM
from datetime import datetime, timezone

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD_ID = os.getenv('DISCORD_GUILD_ID')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

class SignalBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        await self.tree.sync(guild=discord.Object(id=int(GUILD_ID)))
        print(f'Logged in as {self.user}. Slash commands synced to guild.')

client = SignalBot()

def get_latest_prediction(ticker: str):
    from os.path import exists
    dataset_path = f'datasets/{ticker}.pt'
    model_path = f'models/{ticker}.pth'
    if not exists(dataset_path) or not exists(model_path):
        return None, None, f"No model/dataset found for {ticker}"
    X_train, y_train, X_test, y_test = torch.load(dataset_path)
    model = SP500LSTM(X_test.shape[2], 64, 2, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        output = model(X_test[-1].unsqueeze(0))
        probs = F.softmax(output, dim=1).squeeze()
        pred = torch.argmax(probs).item()
        return ("📈 Up" if pred else "📉 Down", probs[pred].item() * 100, None)
    
def get_model_stats(ticker: str):
    from os.path import exists
    dataset_path = f'datasets/{ticker}.pt'
    model_path = f'models/{ticker}.pth'
    if not exists(dataset_path) or not exists(model_path):
        return None, f"No model/dataset found for {ticker}"

    X_train, y_train, X_test, y_test = torch.load(dataset_path)
    model = SP500LSTM(X_test.shape[2], 64, 2, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        outputs = model(X_test)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        correct = (preds == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total

        num_up = (preds == 1).sum().item()
        num_down = (preds == 0).sum().item()

        pred_confidences = probs.max(dim=1).values
        avg_conf = pred_confidences.mean().item()
        min_conf = pred_confidences.min().item()
        max_conf = pred_confidences.max().item()

    return {
        'accuracy': accuracy,
        'samples': total,
        'up': num_up,
        'down': num_down,
        'avg_conf': avg_conf,
        'min_conf': min_conf,
        'max_conf': max_conf
    }, None

    
def fetch_news(ticker: str, api_key: str, limit: int = 5):
    newsapi = NewsApiClient(api_key=api_key)
    result = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=limit)
    articles = result.get('articles', [])
    results = []

    for art in articles:
        title = art.get('title')
        url = art.get('url')
        published = art.get('publishedAt')

        try:
            published_dt = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            time_ago = humanize.naturaltime(datetime.now(timezone.utc) - published_dt)
        except:
            time_ago = "unknown time"

        if title and url:
            results.append(f"• [{title}]({url}) — *{time_ago}*")
    return results


@client.tree.command(name="predict", description="Get the latest signal", guild=discord.Object(id=int(GUILD_ID)))
@app_commands.describe(ticker="Ticker like AAPL, SPY")
async def predict(interaction: discord.Interaction, ticker: str):
    ticker = ticker.upper()
    direction, confidence, error = get_latest_prediction(ticker)
    if error:
        await interaction.response.send_message(f"⚠️ {error}", ephemeral=True)
    else:
        await interaction.response.send_message(f"**{ticker} Signal**\nDirection: {direction}\nConfidence: {confidence:.2f}%")


@client.tree.command(name="news", description="Get the latest financial news for a ticker", guild=discord.Object(id=int(GUILD_ID)))
@app_commands.describe(ticker="Ticker like AAPL, TSLA")
async def news(interaction: discord.Interaction, ticker: str):
    if not NEWSAPI_KEY:
        await interaction.response.send_message("⚠️ News API key not set in .env", ephemeral=True)
        return

    articles = fetch_news(ticker, NEWSAPI_KEY)
    if not articles:
        await interaction.response.send_message(f"No news found for {ticker.upper()}", ephemeral=True)
        return

    await interaction.response.send_message(f"**Latest News for {ticker.upper()}:**\n" + "\n".join(articles))

@client.tree.command(name="stats", description="Get model accuracy for a ticker", guild=discord.Object(id=int(GUILD_ID)))
@app_commands.describe(ticker="Ticker like AAPL, SPY")
async def stats(interaction: discord.Interaction, ticker: str):
    ticker = ticker.upper()
    stats_data, error = get_model_stats(ticker)
    if error:
        await interaction.response.send_message(f"⚠️ {error}", ephemeral=True)
        return

    await interaction.response.send_message(
        f"**{ticker} Model Stats**\n"
        f"📊 Test Samples: {stats_data['samples']}\n"
        f"✅ Accuracy: {stats_data['accuracy']:.2%}\n"
        f"📈 Predicted Up: {stats_data['up']}, 📉 Down: {stats_data['down']}\n"
        f"🎯 Confidence (avg/min/max): "
        f"{stats_data['avg_conf']*100:.2f}% / {stats_data['min_conf']*100:.2f}% / {stats_data['max_conf']*100:.2f}%"
    )


client.run(TOKEN)