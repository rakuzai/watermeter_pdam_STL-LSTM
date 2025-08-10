import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_excel('datafix_watermeter.xlsx', dtype={"nsb": str})
df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.head()

df.info()

df['periode'].min()

df['periode'].max()


duplicates = df[df.duplicated()]

duplicates_all = df[df.duplicated(keep=False)]

print("Number of duplicated rows:", duplicates_all.shape[0])
print(duplicates_all)


df = df.drop_duplicates(keep=False)
df.info()


# Compute IQR
Q1 = df['pakai'].quantile(0.25)
Q3 = df['pakai'].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df = df[(df['pakai'] >= lower_bound) & (df['pakai'] <= upper_bound)]


df = df.sort_values(['nsb', 'periode'])
df.head()


stl = STL(df['pakai'], period=12)
res = stl.fit()

df['trend'] = res.trend
df['seasonal'] = res.seasonal
df['resid'] = res.resid


df['periode'] = pd.to_datetime(df['periode'].astype(str), format='%Y%m')
df = df.sort_values('periode').set_index('periode')


split_date = df.index.max() - pd.DateOffset(years=2)
train = df[df.index <= split_date]
test = df[df.index > split_date]

components = ['trend', 'seasonal', 'resid']


# ==== Helper: buat sequence data ====
def create_sequences(series, seq_len=12):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

# ==== PyTorch LSTM Model ====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # ambil output di time step terakhir
        return out

# ==== Storage ====
preds_test = pd.DataFrame(index=test.index)
metrics = {}

# ==== 3. Training tiap komponen ====
seq_len = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for comp in components:
    # --- Data prep ---
    series_train = train[comp].values.astype(np.float32)
    series_test = test[comp].values.astype(np.float32)

    X_train, y_train = create_sequences(series_train, seq_len)
    X_train = torch.tensor(X_train).unsqueeze(-1).to(device)  # [batch, seq_len, 1]
    y_train = torch.tensor(y_train).unsqueeze(-1).to(device)  # [batch, 1]

    # --- Model ---
    model = LSTMModel(input_size=1, hidden_size=50).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Train ---
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    # --- Predict test (walk-forward) ---
    model.eval()
    history = list(series_train)
    preds = []
    for actual in series_test:
        seq_input = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            yhat = model(seq_input).item()
        preds.append(yhat)
        history.append(actual)  # ganti dengan yhat kalau mau pure forecasting

    preds_test[comp] = preds

    # Evaluate dan print hasil
    metrics[comp] = evaluate(series_test, preds)
    print(f"Metrics untuk komponen {comp}: {metrics[comp]}")

    # --- Evaluate ---
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return {
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': rmse,
            'MAPE': mape
        }
    metrics[comp] = evaluate(series_test, preds)

# Metrics untuk komponen trend: {'R2': -0.757947564125061, 'MAE': 8.836202621459961, 'RMSE': 11.56580136568447, 'MAPE': 17720.857327144477}
# Metrics untuk komponen seasonal: {'R2': 0.006486177444458008, 'MAE': 1.8399001359939575, 'RMSE': 2.602555827074288, 'MAPE': 183.50573268765154}
# Metrics untuk komponen resid: {'R2': 0.007845580577850342, 'MAE': 3.1119842529296875, 'RMSE': 4.55476350540947, 'MAPE': 310.2436387571908}