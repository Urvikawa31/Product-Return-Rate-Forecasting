import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
sys.path.append(os.path.abspath("04_Model_Training"))
sys.path.append(os.path.abspath("."))
from quant_mlp import get_model as get_mlp
from quant_recurrent import get_rnn, get_lstm, get_gru
from quant_tcn import get_tcn
from logger_util import StageTimer

# Paths
FEATURE_PATH = "03_Feature_Extraction/features_return_rate.csv"
MODEL_DIR = "04_Model_Training/model"
OUTPUT_DIR = "05_Model_Validation_Evaluation"
METRICS_JSON = "pipeline_metrics.json"

os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/evaluation_metrics", exist_ok=True)

# Business Parameters
AVG_ITEM_PRICE = 50.0
RETURN_COST_PCT = 0.15 
PIPELINE_INVESTMENT = 1000.0
# SYNCED QUANTILES for Target ROI (Final Optimized Polish)
QUANTILES = [0.05, 0.5, 0.95]

def load_conformal_data_with_cats():
    df = pd.read_csv(FEATURE_PATH)
    cat_cols = [c for c in df.columns if c.startswith('Cat_')]
    df['Category_Label'] = df[cat_cols].idxmax(axis=1).str.replace('Cat_', '')
    drop_cols = ['Week', 'ReturnRate', 'TotalOrders', 'ReturnCount', 'WeekOfYear', 'Category_Label']
    x_cols = [c for c in df.columns if c not in drop_cols]
    X, y, cats = df[x_cols].values, df['ReturnRate'].values, df['Category_Label'].values
    split_idx = int(len(df) * 0.8)
    X_full_test, y_full_test, cats_full_test = X[split_idx:], y[split_idx:], cats[split_idx:]
    y_train_actuals = y[:split_idx]
    scaler_x = MinMaxScaler()
    scaler_x.fit(X[:split_idx])
    X_full_test = scaler_x.transform(X_full_test)
    test_split = int(len(X_full_test) * 0.5)
    X_eval, y_eval, cats_eval = X_full_test[test_split:], y_full_test[test_split:], cats_full_test[test_split:]
    X_cal, y_cal = X_full_test[:test_split], y_full_test[:test_split]
    def prepare_tensor(X):
        return torch.FloatTensor(X.astype(float)), torch.FloatTensor(X.reshape(X.shape[0], 1, X.shape[1]).astype(float))
    X_cal_flat, X_cal_seq = prepare_tensor(X_cal)
    X_eval_flat, X_eval_seq = prepare_tensor(X_eval)
    return (X_cal_flat, X_cal_seq, torch.FloatTensor(y_cal.astype(float))), \
           (X_eval_flat, X_eval_seq, torch.FloatTensor(y_eval.astype(float)), cats_eval), \
           y_train_actuals, len(x_cols)

def pinball_loss(y_true, y_pred, q):
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))

def calculate_conformal_adjustment(y_cal, preds_cal, alpha=0.1):
    y_cal = y_cal.flatten()
    L, U = preds_cal[:, 0], preds_cal[:, 2]
    E = np.maximum(L - y_cal, y_cal - U)
    q_val = np.quantile(E, np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal))
    return q_val

def apply_and_evaluate(y_true, preds_raw, q_val, quantiles=QUANTILES):
    y_true = y_true.flatten()
    adj_L, adj_M, adj_U = preds_raw[:, 0] - q_val, preds_raw[:, 1], preds_raw[:, 2] + q_val
    preds_adj = np.stack([adj_L, adj_M, adj_U], axis=1)
    picp = np.mean((y_true >= adj_L) & (y_true <= adj_U))
    pinaw = np.mean(adj_U - adj_L) / (y_true.max() - y_true.min() + 1e-6)
    pb_loss = np.mean([pinball_loss(y_true, preds_adj[:, i], q) for i, q in enumerate(quantiles)])
    winkler = np.mean((adj_U - adj_L) + (2/0.1) * ((adj_L - y_true) * (y_true < adj_L) + (y_true - adj_U) * (y_true > adj_U)))
    return {"Pinball Loss": pb_loss, "Coverage (PICP)": picp, "Winkler Score": winkler, "Interval Width": pinaw}

def calculate_business_utility(df_row):
    units, cost = 100, AVG_ITEM_PRICE * RETURN_COST_PCT
    avoided_loss = units * cost * df_row["Coverage (PICP)"] * (1 - df_row["Interval Width"] * 0.5)
    roi = ((avoided_loss * 52) - PIPELINE_INVESTMENT) / PIPELINE_INVESTMENT * 100
    return pd.Series([avoided_loss, roi])

def individual_forecast_plot(name, y_true, preds_adj):
    plt.figure(figsize=(14, 6))
    time_steps = np.arange(len(y_true))
    L, M, U = preds_adj[:, 0], preds_adj[:, 1], preds_adj[:, 2]
    plt.plot(time_steps, y_true, 'k-o', label='Actual Return Rate', alpha=0.8, markersize=4)
    plt.plot(time_steps, M, 'b--', label='Predicted Median', alpha=0.7)
    plt.fill_between(time_steps, L, U, color='blue', alpha=0.2, label='90% Calibrated Interval')
    plt.title(f"Figure 6: {name} Forecasting Interval (Optimized)")
    plt.xlabel("Weeks (Evaluation Period)")
    plt.ylabel("Return Rate")
    plt.ylim(0, 1.2)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/Figure_6_{name}_Forecast_Interval.png")
    plt.close()

def plot_full_horizon_forecast(name, y_train_full, y_eval, preds_adj):
    plt.figure(figsize=(18, 7))
    full_actuals, split_point = np.concatenate([y_train_full, y_eval]), len(y_train_full)
    time_steps = np.arange(len(full_actuals))
    plt.plot(time_steps[:split_point], y_train_full, color='black', alpha=0.4, label='Training History')
    plt.plot(time_steps[split_point:], y_eval, 'k-o', label='Future Actuals', markersize=4)
    L, M, U = preds_adj[:, 0], preds_adj[:, 1], preds_adj[:, 2]
    plt.plot(time_steps[split_point:], M, 'b--', label='Forecasted Median', alpha=0.9)
    plt.fill_between(time_steps[split_point:], L, U, color='blue', alpha=0.2, label='90% Confidence Interval')
    plt.axvline(x=split_point, color='red', linestyle=':', label='Forecast Start')
    plt.title(f"Figure 9: {name} Full-Horizon Forecasting (Final Polish)", fontsize=16, fontweight='bold')
    plt.ylim(0, 1.2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/Figure_9_{name}_Full_Horizon_Forecast.png", dpi=300)
    plt.close()

def run_evaluation():
    (X_cal_flat, X_cal_seq, y_cal), (X_eval_flat, X_eval_seq, y_eval, cats_eval), y_train_full, input_dim = load_conformal_data_with_cats()
    model_configs = [("MLP", get_mlp(input_dim, 3), X_cal_flat, X_eval_flat), 
                     ("RNN", get_rnn(input_dim, 3), X_cal_seq, X_eval_seq), 
                     ("LSTM", get_lstm(input_dim, 3), X_cal_seq, X_eval_seq), 
                     ("GRU", get_gru(input_dim, 3), X_cal_seq, X_eval_seq), 
                     ("TCN", get_tcn(input_dim, 3), X_cal_seq, X_eval_seq)]
    
    y_eval_np, results_cal, cat_results = y_eval.numpy(), [], []
    for name, model, cal_data, eval_data in model_configs:
        path = f"{MODEL_DIR}/{name.lower()}_pinball.pth"
        if not os.path.exists(path): continue
        model.load_state_dict(torch.load(path))
        model.eval()
        with torch.no_grad():
            preds_cal, preds_eval = model(cal_data).numpy(), model(eval_data).numpy()
            
        # Calibration & Evaluation
        q_val = calculate_conformal_adjustment(y_cal.numpy(), preds_cal)
        metrics = apply_and_evaluate(y_eval_np, preds_eval, q_val)
        
        # Plotting
        adj_L, adj_M, adj_U = preds_eval[:, 0]-q_val, preds_eval[:, 1], preds_eval[:, 2]+q_val
        preds_adj = np.stack([adj_L, adj_M, adj_U], axis=1)
        individual_forecast_plot(name, y_eval_np, preds_adj)
        plot_full_horizon_forecast(name, y_train_full, y_eval_np, preds_adj)
        
        results_cal.append({"Model": name, "Conformal_Adj_Q": q_val, **metrics})
        
        # Granular Analysis
        for cat in np.unique(cats_eval):
            m = (cats_eval == cat)
            y_c, L_c, U_c = y_eval_np[m], adj_L[m], adj_U[m]
            cat_results.append({"Model": name, "Category": cat, "Coverage (PICP)": np.mean((y_c>=L_c)&(y_c<=U_c))})
            
    res_df = pd.DataFrame(results_cal)
    # Business ROI Logic
    res_df[["Avoided Weekly Loss ($)", "Annualized ROI (%)"]] = res_df.apply(calculate_business_utility, axis=1)
    res_df.round(4).to_csv(f"{OUTPUT_DIR}/evaluation_metrics/Final_Optimized_Benchmarking.csv", index=False)
    
    # Dashboards
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    palette = "viridis"
    sns.barplot(x="Model", y="Coverage (PICP)", data=res_df, ax=axes[0, 0], palette=palette)
    axes[0, 0].axhline(0.9, color='red', linestyle='--')
    axes[0, 0].set_title("1. Optimized Reliability (PICP)", fontweight='bold')
    sns.barplot(x="Model", y="Winkler Score", data=res_df, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("2. Sharpness Fidelity (Winkler Score)", fontweight='bold')
    sns.barplot(x="Model", y="Avoided Weekly Loss ($)", data=res_df, ax=axes[1, 0], palette="magma")
    axes[1, 0].set_title("3. Potential Weekly Savings", fontweight='bold')
    sns.barplot(x="Model", y="Annualized ROI (%)", data=res_df, ax=axes[1, 1], palette="magma")
    axes[1, 1].set_title("4. Optimized Forecast ROI%", fontweight='bold')
    plt.suptitle("Figure 5: Final Optimized Performance Dashboard", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{OUTPUT_DIR}/plots/Figure_5_Optimized_Dashboard.png", dpi=300)
    plt.close()
    
    # Category Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pd.DataFrame(cat_results).pivot(index="Category", columns="Model", values="Coverage (PICP)"), annot=True, cmap="YlGnBu")
    plt.title("Figure 8: Final Optimized Category Reliability", fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/plots/Figure_8_Optimized_Heatmap.png", dpi=300)
    plt.close()

    print("--- Final Optimized Evaluation Complete ---")
    print(res_df[["Model", "Coverage (PICP)", "Winkler Score", "Annualized ROI (%)"]])

if __name__ == "__main__":
    run_evaluation()
