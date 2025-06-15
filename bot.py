import os
import torch
import torch.nn.functional as F
import discord
from discord import app_commands
from dotenv import load_dotenv
from scripts.model import SP500LSTM

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD_ID = os.getenv('DISCORD_GUILD_ID')

HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 2

def get_latest_prediction(ticker: str):
    dataset_path = os.path.join('datasets', f'{ticker}.pt')
    model_path = os.path.join('models', f'{ticker}.pth')

    if not os.path.exists(dataset_path) or not os.path.exists(model_path):
        return None, None, f"No model or dataset found for ticker '{ticker}'."

    try:
        X_train, y_train, X_test, y_test = torch.load(dataset_path)
        input_size = X_test.shape[2]

        model = SP500LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            latest_input = X_test[-1].unsqueeze(0)
            output = model(latest_input)
            probs = F.softmax(output, dim=1).squeeze()
            prediction = torch.argmax(probs).item()
            direction = "📈 Up" if prediction == 1 else "📉 Down"
            confidence = probs[prediction].item() * 100
        return direction, confidence, None
    except Exception as e:
        return None, None, f"Error running prediction for {ticker}: {e}"


class SignalBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        guild = discord.Object(id=int(GUILD_ID))
        await self.tree.sync(guild=guild)
        print(f'Slash commands synced to guild {GUILD_ID}')

client = SignalBot()

@client.tree.command(
    name="predict",
    description="Get the latest market signal prediction",
    guild=discord.Object(id=int(GUILD_ID))
)
@app_commands.describe(ticker="Stock ticker symbol (e.g. SPY, AAPL)")
async def predict_command(interaction: discord.Interaction, ticker: str):
    ticker = ticker.upper()
    direction, confidence, error = get_latest_prediction(ticker)

    if error:
        await interaction.response.send_message(f"⚠️ {error}", ephemeral=True)
    else:
        await interaction.response.send_message(
            f'**Latest {ticker} Signal**\nDirection: {direction}\nConfidence: {confidence:.2f}%'
        )

client.run(TOKEN)