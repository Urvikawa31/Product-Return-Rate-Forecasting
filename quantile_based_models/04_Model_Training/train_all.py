import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import time
import sys
sys.path.append(os.path.abspath("."))
from logger_util import StageTimer
from quant_mlp import get_model as get_mlp
from quant_recurrent import get_rnn, get_lstm, get_gru
from quant_tcn import get_tcn

# Configuration
INPUT_PATH = "03_Feature_Extraction/features_return_rate.csv"
MODEL_SAVE_DIR = "04_Model_Training/model"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 200 
SPIKE_WEIGHT = 5.0 
# OPTIMIZED QUANTILES for Peak ROI (Restored Flagship Sharpness)
QUANTILES = [0.05, 0.5, 0.95] 

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def prepare_data():
    df = pd.read_csv(INPUT_PATH)
    drop_cols = ['Week', 'ReturnRate', 'TotalOrders', 'ReturnCount', 'WeekOfYear']
    x_cols = [c for c in df.columns if c not in drop_cols]
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train = torch.FloatTensor(train_df[x_cols].values.astype(float))
    y_train = torch.FloatTensor(train_df['ReturnRate'].values.astype(float))
    X_test = torch.FloatTensor(test_df[x_cols].values.astype(float))
    y_test = torch.FloatTensor(test_df['ReturnRate'].values.astype(float))
    X_tr_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_te_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, y_train, X_test, y_test, X_tr_seq, X_te_seq, len(x_cols)

def weighted_pinball_loss(preds, target, quantiles, weight_lambda=5.0):
    """
    Final Polish: Optimized Loss with Balancing Penalty
    """
    total_loss = 0
    weights = 1.0 + weight_lambda * target
    for i, q in enumerate(quantiles):
        err = target - preds[:, i].unsqueeze(1)
        loss = torch.max(q * err, (q - 1) * err)
        total_loss += (weights * loss).mean()
    
    # --- Softened Quantile Crossing Penalty (Restored ROI sharpness) ---
    crossing_penalty = torch.relu(preds[:, 0] - preds[:, 1]).mean() + \
                       torch.relu(preds[:, 1] - preds[:, 2]).mean()
    
    return (total_loss / len(quantiles)) + crossing_penalty * 1.0

def train_model(model, train_loader, name):
    print(f"--- Final Polish: Training {name} ---")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    with StageTimer(f"Training {name}"):
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_x)
                loss = weighted_pinball_loss(preds, batch_y.unsqueeze(1), QUANTILES, SPIKE_WEIGHT)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            if (epoch+1) % 40 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.6f}")
    
    torch.save(model.state_dict(), f"{MODEL_SAVE_DIR}/{name.lower()}_pinball.pth")

if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, X_tr_seq, X_te_seq, input_dim = prepare_data()
    train_loader_flat = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    train_loader_seq = DataLoader(TensorDataset(X_tr_seq, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    
    models = [
        (get_mlp(input_dim, 3), train_loader_flat, "MLP"),
        (get_rnn(input_dim, 3), train_loader_seq, "RNN"),
        (get_lstm(input_dim, 3), train_loader_seq, "LSTM"),
        (get_gru(input_dim, 3), train_loader_seq, "GRU"),
        (get_tcn(input_dim, 3), train_loader_seq, "TCN")
    ]
    
    for model, loader, name in models:
        train_model(model, loader, name)
    print("--- Final Model Polish Complete ---")
