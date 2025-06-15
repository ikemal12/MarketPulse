import discord
import torch
import torch.nn.functional as F
from scripts.model import SP500LSTM
from dotenv import load_dotenv
import os

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

X_train, y_train, X_test, y_test = torch.load('dataset.pt')
INPUT_SIZE = X_test.shape[2]
model = SP500LSTM(INPUT_SIZE, hidden_size=64, num_layers=2, output_size=2)
model.load_state_dict(torch.load('sp500_lstm.pth'))
model.eval()

with torch.no_grad():
    latest_input = X_test[-1].unsqueeze(0)  
    output = model(latest_input)
    probs = F.softmax(output, dim=1).squeeze()
    prediction = torch.argmax(probs).item()
    direction = "📈 Up" if prediction == 1 else "📉 Down"
    confidence = probs[prediction].item() * 100

intents = discord.Intents.default()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    channel = discord.utils.get(client.get_all_channels(), name='general')
    if channel:
        await channel.send(
            f'**Daily S&P 500 Signal**\n'
            f'Direction: {direction}\n'
            f'Confidence: {confidence:.2f}%'
        )
    await client.close()

client.run(TOKEN)