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
TICKER = 'SPY'

DATASET_PATH = os.path.join('datasets', f'{TICKER}.pt')
MODEL_PATH = os.path.join('models', f'{TICKER}.pth')

X_train, y_train, X_test, y_test = torch.load(DATASET_PATH)
INPUT_SIZE = X_test.shape[2]
model = SP500LSTM(INPUT_SIZE, hidden_size=64, num_layers=2, output_size=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def get_latest_prediction():
    with torch.no_grad():
        latest_input = X_test[-1].unsqueeze(0)
        output = model(latest_input)
        probs = F.softmax(output, dim=1).squeeze()
        prediction = torch.argmax(probs).item()
        direction = '📈 Up' if prediction == 1 else '📉 Down
        confidence = probs[prediction].item() * 100
    return direction, confidence

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

@client.tree.command(name='predict', description='Get the latest market signal prediction', guild=discord.Object(id=int(GUILD_ID)))
async def predict_command(interaction: discord.Interaction):
    direction, confidence = get_latest_prediction()
    await interaction.response.send_message(
        f'**Latest {TICKER} Signal**\nDirection: {direction}\nConfidence: {confidence:.2f}%'
    )

client.run(TOKEN)