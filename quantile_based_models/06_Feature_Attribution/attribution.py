import torch
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath("04_Model_Training"))
from quant_mlp import get_model as get_mlp
from model.quantile_losses import PinballLoss

# Paths
FEATURE_PATH = "03_Feature_Extraction/features_return_rate.csv"
MODEL_PATH = "04_Model_Training/model/mlp_pinball.pth"
OUTPUT_DIR = "06_Feature_Attribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(FEATURE_PATH)
    drop_cols = ['Week', 'ReturnRate', 'TotalOrders', 'ReturnCount', 'WeekOfYear']
    x_cols = [c for c in df.columns if c not in drop_cols]
    y_col = 'ReturnRate'
    
    X = df[x_cols].values
    y = df[y_col].values
    split_idx = int(len(df) * 0.8)
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    scaler_x = MinMaxScaler()
    scaler_x.fit(X[:split_idx])
    X_test = scaler_x.transform(X_test)
    
    return torch.FloatTensor(X_test), torch.FloatTensor(y_test), x_cols

def permutation_importance(model, X, y, x_cols):
    model.eval()
    criterion = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    
    with torch.no_grad():
        baseline_preds = model(X)
        baseline_loss = criterion(baseline_preds, y.unsqueeze(1)).item()
    
    importances = {}
    for i, col in enumerate(x_cols):
        X_perm = X.clone()
        # Shuffle the feature
        perm = torch.randperm(X.shape[0])
        X_perm[:, i] = X[perm, i]
        
        with torch.no_grad():
            perm_preds = model(X_perm)
            perm_loss = criterion(perm_preds, y.unsqueeze(1)).item()
        
        # Importance is the increase in loss
        importances[col] = (perm_loss - baseline_loss) / baseline_loss
    
    return importances

def run_attribution():
    X, y, x_cols = load_data()
    input_dim = len(x_cols)
    
    model = get_mlp(input_dim, 3)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded.")
    else:
        print("Model not found.")
        return

    importances = permutation_importance(model, X, y, x_cols)
    
    imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
    imp_df = imp_df.sort_values('Importance', ascending=False)
    imp_df.to_csv(f"{OUTPUT_DIR}/feature_importance_mlp.csv", index=False)
    
    print("Feature attribution complete. Results saved.")
    print(imp_df.head(10))

if __name__ == "__main__":
    run_attribution()
