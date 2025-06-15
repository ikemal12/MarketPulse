import pandas as pd
import numpy as np  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

SEQUENCE_LENGTH = 20
FEATURE_COLUMNS = ['Close', 'High', 'Low', 'Open', 'Volume', 'Return']

df = pd.read_csv('sp500.csv', header=0, skiprows=[1, 2])
df.columns = df.columns.str.strip()
df = df.drop(columns=['Price'], errors='ignore')
df = df.dropna()
print(df.columns.tolist())

scaler = StandardScaler()
features = scaler.fit_transform(df[FEATURE_COLUMNS])

scaler_params = {
    'mean': scaler.mean_,
    'scale': scaler.scale_
}
np.save('scaler.npy', scaler_params)

x = []
y = []

for i in range(SEQUENCE_LENGTH, len(features)):
    x.append(features[i - SEQUENCE_LENGTH:i])
    y.append(df['Target'].iloc[i])

x = np.array(x)
y = np.array(y)

print(f'Sequences created: {x.shape}, Labels: {y.shape}')

# split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

torch.save((X_train, y_train, X_test, y_test), 'dataset.pt')
print('Dataset saved')