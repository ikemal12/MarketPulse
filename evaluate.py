import torch
from model import SP500LSTM
import torch.nn.functional as F

X_train, y_train, X_test, y_test = torch.load('dataset.pt')

INPUT_SIZE = X_train.shape[2]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 2

model = SP500LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load('sp500_lstm.pth'))
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)

    correct = (predictions == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total 

print(f'Test Accuracy: {accuracy:.2%}')

