import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import SP500LSTM
from torch.optim.lr_scheduler import StepLR

def main(ticker):
    dataset_path = os.path.join('datasets', f'{ticker}.pt')
    model_path = os.path.join('models', f'{ticker}.pth')

    X_train, y_train, X_test, y_test = torch.load(dataset_path)

    INPUT_SIZE = X_train.shape[2]
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = 2
    EPOCHS = 30
    BATCH_SIZE = 32
    LR = 0.001

    model = SP500LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

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
        scheduler.step()

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    args = parser.parse_args()
    main(args.ticker.upper())
