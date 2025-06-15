import torch
import torch.nn as nn
import torch.optim as optim
from model import SP500LSTM

X_train, y_train, X_test, y_test = torch.load('dataset.pt')

INPUT_SIZE = X_train.shape[2]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 2
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

model = SP500LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss:.4f}')

torch.save(model.state_dict(), 'sp500_lstm.pth')
print('Model saved')
