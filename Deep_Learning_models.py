#!/usr/bin/env python
# coding: utf-8

# # Product Return Rate Forecasting using Deep Learning
# ## MLP | RNN | LSTM | GRU | TCN | Stacked Ensemble
# 
# ---
# 
# **Objective:** Predict weekly product return rates for the next 4 weeks, then identify the Top 5 products driving those returns.
# 
# ### Pipeline Flow
# 
# | Step | Section | Description |
# |------|---------|-------------|
# | 1 | Data Loading & Cleaning | Load and prepare the UCI Online Retail II dataset |
# | 2 | Feature Engineering | Create return flag and weekly aggregation |
# | 3 | EDA | Visual overview of return rate trends |
# | 4 | Preprocessing | MinMax scaling and sliding-window sequences |
# | 5 | Train / Test Split | Chronological 80/20 split |
# | 6 | Model Training | Train MLP, RNN, LSTM, GRU, TCN + Stacked Ensemble |
# | 7 | Evaluation | Compare all models on MSE, RMSE, MAE, MAPE |
# | **8** | **4-Week Forecast** | **Predict return rate for the next 4 weeks** |
# | **9** | **Top 5 Return Products** | **Which products are driving those returns?** |
# 

# ## 0. Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time
warnings.filterwarnings('ignore')

np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (SimpleRNN, LSTM, GRU, Dense, Flatten,
                                      Dropout, BatchNormalization, Input, concatenate)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tcn import TCN

plt.style.use('seaborn-v0_8-darkgrid')
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
plt.rcParams.update({'figure.dpi': 100, 'font.size': 12,
                     'axes.titlesize': 14, 'axes.labelsize': 12})

print("All libraries loaded successfully.")
print(f"  TensorFlow : {tf.__version__}")
print(f"  NumPy      : {np.__version__}")
print(f"  Pandas     : {pd.__version__}")


# ## 1. Data Loading & Cleaning
# 
# - Load CSV with `encoding='latin1'`
# - Drop null rows, convert dates, remove zero-quantity adjustments
# 

# In[2]:


print("Loading dataset ...")
t0 = time.time()
df_raw = pd.read_csv('online_retail_II.csv', encoding='latin1')
print(f"Loaded in {time.time()-t0:.1f}s  |  Shape: {df_raw.shape}")

print("\n-- Missing values --")
null_summary = df_raw.isnull().sum().to_frame('nulls')
null_summary['pct'] = (null_summary['nulls'] / len(df_raw) * 100).round(2)
print(null_summary[null_summary['nulls'] > 0])

df = df_raw.copy()
df.dropna(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df[df['Quantity'] != 0]

print(f"\nAfter cleaning  |  Shape: {df.shape}")
print(f"Date range : {df['InvoiceDate'].min().date()}  to  {df['InvoiceDate'].max().date()}")
print(f"Unique invoices:   {df['Invoice'].nunique():,}")
print(f"Unique products:   {df['StockCode'].nunique():,}")
print(f"Unique customers:  {df['Customer ID'].nunique():,}")


# ## 2. Feature Engineering
# 
# Invoice starting with **'C'** = cancellation/return → `is_returned = 1`.  
# Aggregate by **ISO week** to build the weekly return-rate time series.
# 

# In[3]:


df['is_returned'] = df['Invoice'].astype(str).str.startswith('C').astype(int)
df['Week'] = df['InvoiceDate'].dt.to_period('W').dt.start_time

n_returns = df['is_returned'].sum()
print(f"Total transactions : {len(df):,}")
print(f"Return transactions: {n_returns:,}  ({n_returns/len(df)*100:.2f}%)")
print(f"Normal transactions: {len(df)-n_returns:,}")


# ## 3. Exploratory Data Analysis

# In[4]:


weekly = (df.groupby('Week')['is_returned']
            .agg(['mean', 'sum', 'count'])
            .rename(columns={'mean':'return_rate',
                             'sum':'n_returns',
                             'count':'n_transactions'})
            .sort_index()
            .reset_index())

print(f"Weekly data points : {len(weekly)}")
print("\n-- Return-rate statistics --")
print(weekly['return_rate'].describe().round(4))

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
fig.suptitle('Weekly Return Rate -- Overview', fontsize=16, y=1.01)

ax = axes[0]
ax.plot(weekly['Week'], weekly['return_rate'], color=PALETTE[0],
        linewidth=1.5, marker='o', markersize=3, label='Weekly Return Rate')
ax.fill_between(weekly['Week'], weekly['return_rate'], alpha=0.15, color=PALETTE[0])
ax.axhline(weekly['return_rate'].mean(), color='red', linestyle='--',
           linewidth=1.2, label=f"Mean = {weekly['return_rate'].mean():.3f}")
ax.set_title('A) Weekly Return Rate Over Time')
ax.set_ylabel('Return Rate')
ax.legend(loc='upper left')

ax = axes[1]
ax.bar(weekly['Week'], weekly['n_transactions'], color=PALETTE[1],
       width=5, alpha=0.7, label='Total Transactions')
ax.set_title('B) Weekly Transaction Volume')
ax.set_ylabel('Transaction Count')
ax.legend()

ax = axes[2]
ax.hist(weekly['return_rate'], bins=25, color=PALETTE[2],
        edgecolor='white', alpha=0.8, density=True)
ax.axvline(weekly['return_rate'].mean(), color='red', linestyle='--',
           linewidth=1.5, label='Mean')
ax.axvline(weekly['return_rate'].median(), color='orange', linestyle=':',
           linewidth=1.5, label='Median')
ax.set_title('C) Distribution of Weekly Return Rate')
ax.set_xlabel('Return Rate')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
plt.savefig('fig01_eda_overview.png', bbox_inches='tight')
plt.close()


# ## 4. Data Preprocessing
# 
# - **MinMax scaling** to [0, 1]
# - **Sliding window:** 8 weeks input → predict next 4 weeks (multi-output)
# 

# In[5]:


WINDOW    = 8   # look-back weeks
OUT_STEPS = 4   # predict next 4 weeks

scaler = MinMaxScaler()
values = weekly['return_rate'].values.reshape(-1, 1)
scaled = scaler.fit_transform(values)

print(f"Original range : [{values.min():.4f}, {values.max():.4f}]")
print(f"Scaled   range : [{scaled.min():.4f}, {scaled.max():.4f}]")

X, y = [], []
for i in range(len(scaled) - WINDOW - OUT_STEPS + 1):
    X.append(scaled[i : i + WINDOW])
    y.append(scaled[i + WINDOW : i + WINDOW + OUT_STEPS].flatten())

X = np.array(X)
y = np.array(y)

print(f"\nSequence shapes:  X = {X.shape}  |  y = {y.shape}")
print(f"Look-back window : {WINDOW} weeks")
print(f"Forecast horizon : {OUT_STEPS} weeks ahead")


# ## 5. Train / Test Split (80/20 Chronological)

# In[6]:


split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples : {len(X_train)}")
print(f"Test     samples : {len(X_test)}")

split_date = weekly['Week'].iloc[split + WINDOW + OUT_STEPS - 2]

fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(weekly['Week'], weekly['return_rate'],
        color='steelblue', linewidth=1.5, label='All data')
ax.axvline(split_date, color='crimson', linestyle='--',
           linewidth=2, label=f'Train/Test split ({split_date.date()})')
ax.fill_betweenx([0, weekly['return_rate'].max() * 1.1],
                  weekly['Week'].iloc[0], split_date,
                  alpha=0.08, color='green', label='Train region')
ax.fill_betweenx([0, weekly['return_rate'].max() * 1.1],
                  split_date, weekly['Week'].iloc[-1],
                  alpha=0.08, color='crimson', label='Test region')
ax.set_title('Train / Test Split -- Weekly Return Rate')
ax.set_ylabel('Return Rate')
ax.legend()
plt.tight_layout()
plt.savefig('fig02_train_test_split.png', bbox_inches='tight')
plt.close()


# ## 6. Model Building & Training
# 
# | Model | Type | Architecture |
# |-------|------|-------------|
# | **MLP** | Baseline (no temporal awareness) | Flatten → Dense(64→32→16) → Dense(4) |
# | **SimpleRNN** | Recurrent | 2-layer stacked (64→32) → Dense(4) |
# | **LSTM** | Recurrent (gated) | 2-layer stacked (64→32) → Dense(4) |
# | **GRU** | Recurrent (gated) | 2-layer stacked (64→32) → Dense(4) |
# | **TCN** | Convolutional | 64 filters, k=3, dilations=[1,2,4] → Dense(4) |
# | **Ensemble** | Stacked meta-learner | Combines all 5 base model outputs → Dense(4) |
# 
# All models: Dropout=0.2, BatchNorm, Adam(lr=1e-3), EarlyStopping(patience=15)
# 

# In[7]:


EPOCHS     = 100
BATCH_SIZE = 8
INPUT_SHAPE = (WINDOW, 1)
LR          = 1e-3

callbacks_common = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=0),
]

# ── Model Builders ───────────────────────────────────────────────
def build_mlp(input_shape, out_steps):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(out_steps),
    ], name='MLP')
    model.compile(optimizer=Adam(LR), loss='mse')
    return model

def build_rnn(input_shape, out_steps):
    model = Sequential([
        SimpleRNN(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        SimpleRNN(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(out_steps),
    ], name='SimpleRNN')
    model.compile(optimizer=Adam(LR), loss='mse')
    return model

def build_lstm(input_shape, out_steps):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(out_steps),
    ], name='LSTM')
    model.compile(optimizer=Adam(LR), loss='mse')
    return model

def build_gru(input_shape, out_steps):
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(out_steps),
    ], name='GRU')
    model.compile(optimizer=Adam(LR), loss='mse')
    return model

def build_tcn(input_shape, out_steps):
    inp = Input(shape=input_shape)
    x = TCN(nb_filters=64, kernel_size=3,
            dilations=[1, 2, 4],
            dropout_rate=0.2,
            return_sequences=False)(inp)
    x = Dense(16, activation='relu')(x)
    out = Dense(out_steps)(x)
    model = Model(inp, out, name='TCN')
    model.compile(optimizer=Adam(LR), loss='mse')
    return model

BUILDERS = {
    'MLP' : build_mlp,
    'RNN' : build_rnn,
    'LSTM': build_lstm,
    'GRU' : build_gru,
    'TCN' : build_tcn,
}

print("All model builders defined.")
print(f"Models to train: {', '.join(BUILDERS.keys())} + Stacked Ensemble")
print(f"\nInput shape  : {INPUT_SHAPE}  ({WINDOW} weeks, 1 feature)")
print(f"Output shape : ({OUT_STEPS},)  ({OUT_STEPS} weeks ahead)")


# ### 6.1 Training All Models

# In[8]:


trained_models   = {}
training_history = {}

fig, axes = plt.subplots(2, 4, figsize=(26, 10))
fig.suptitle('Training vs Validation Loss -- All Models', fontsize=16)
axes = axes.flatten()

for idx, (name, builder) in enumerate(BUILDERS.items()):
    print(f"\n{'='*50}")
    print(f"  Training {name} ...")
    print('='*50)

    model = builder(INPUT_SHAPE, OUT_STEPS)
    t_start = time.time()

    hist = model.fit(
        X_train, y_train,
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        validation_data = (X_test, y_test),
        callbacks       = callbacks_common,
        verbose         = 0,
    )

    elapsed  = time.time() - t_start
    best_val = min(hist.history['val_loss'])
    stopped  = len(hist.history['loss'])

    print(f"  Stopped at epoch {stopped}/{EPOCHS} | "
          f"Best val_loss = {best_val:.6f} | "
          f"Time = {elapsed:.1f}s")

    trained_models[name]   = model
    training_history[name] = hist

    ax = axes[idx]
    ax.plot(hist.history['loss'],     label='Train loss', color=PALETTE[0], linewidth=1.5)
    ax.plot(hist.history['val_loss'], label='Val loss',   color=PALETTE[3],
            linestyle='--', linewidth=1.5)
    ax.set_title(f'{name} -- Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()

# ── Stacked Ensemble ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("  Training Stacked Ensemble (Meta-learner) ...")
print('='*50)

for m in trained_models.values():
    m.trainable = False

ensemble_input = Input(shape=INPUT_SHAPE)
outputs = [model(ensemble_input) for model in trained_models.values()]
concat = concatenate(outputs) if len(outputs) > 1 else outputs[0]
x = Dense(32, activation='relu')(concat)
ensemble_output = Dense(OUT_STEPS)(x)

ensemble_model = Model(inputs=ensemble_input, outputs=ensemble_output, name='Stacked_Ensemble')
ensemble_model.compile(optimizer=Adam(LR), loss='mse')

t_start = time.time()
hist_ens = ensemble_model.fit(
    X_train, y_train,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    validation_data = (X_test, y_test),
    callbacks       = callbacks_common,
    verbose         = 0,
)
elapsed  = time.time() - t_start
best_val = min(hist_ens.history['val_loss'])
stopped  = len(hist_ens.history['loss'])

print(f"  Stopped at epoch {stopped}/{EPOCHS} | Best val_loss = {best_val:.6f} | Time = {elapsed:.1f}s")

trained_models['Ensemble'] = ensemble_model
training_history['Ensemble'] = hist_ens

ax = axes[5]
ax.plot(hist_ens.history['loss'],     label='Train loss', color=PALETTE[0], linewidth=1.5)
ax.plot(hist_ens.history['val_loss'], label='Val loss',   color=PALETTE[3],
        linestyle='--', linewidth=1.5)
ax.set_title('Ensemble -- Learning Curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
ax.legend()

axes[6].axis('off')
axes[7].axis('off')

plt.tight_layout()
plt.savefig('fig03_learning_curves.png', bbox_inches='tight')
plt.close()
print("\nAll models trained successfully.")


# ## 7. Model Evaluation & Comparison
# 
# Metrics computed in the **original return-rate scale** (inverse-transformed):
# 
# | Metric | Interpretation |
# |--------|---------------|
# | **MSE** | Penalises large errors more |
# | **NMSE** | Normalized Mean Squared Error |
# | **OMC** | Order Management Cost (Abs Err in Returns × Avg Cost) |
# | **RMSE** | Same unit as target |
# | **MAE** | Average absolute deviation |
# | **MAPE** | Percentage error (scale-independent) |
# 

# In[9]:


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def inv_multistep(arr_scaled):
    flat = arr_scaled.reshape(-1, 1)
    inv  = scaler.inverse_transform(flat)
    return inv.reshape(arr_scaled.shape[0], OUT_STEPS)

results     = {}
predictions = {}
y_test_inv  = inv_multistep(y_test)

avg_weekly_txns = weekly['n_transactions'].mean()
avg_cost = df['Price'].mean()

for name, model in trained_models.items():
    preds_scaled = model.predict(X_test, verbose=0)
    preds_inv    = inv_multistep(preds_scaled)
    predictions[name] = preds_inv

    mse_val  = mean_squared_error(y_test_inv.flatten(), preds_inv.flatten())
    nmse_val = mse_val / np.var(y_test_inv.flatten())
    rmse_val = np.sqrt(mse_val)
    mae_val  = mean_absolute_error(y_test_inv.flatten(), preds_inv.flatten())
    mape_val = mape(y_test_inv.flatten(), preds_inv.flatten())
    omc_val  = mae_val * avg_weekly_txns * avg_cost

    rmse_w1 = np.sqrt(mean_squared_error(y_test_inv[:,0], preds_inv[:,0]))
    rmse_w2 = np.sqrt(mean_squared_error(y_test_inv[:,1], preds_inv[:,1]))
    rmse_w3 = np.sqrt(mean_squared_error(y_test_inv[:,2], preds_inv[:,2]))
    rmse_w4 = np.sqrt(mean_squared_error(y_test_inv[:,3], preds_inv[:,3]))

    results[name] = {
        'MSE'      : round(mse_val,  6),
        'NMSE'     : round(nmse_val, 6),
        'OMC'      : round(omc_val,  2),
        'RMSE'     : round(rmse_val, 6),
        'RMSE_W1'  : round(rmse_w1,  6),
        'RMSE_W2'  : round(rmse_w2,  6),
        'RMSE_W3'  : round(rmse_w3,  6),
        'RMSE_W4'  : round(rmse_w4,  6),
        'MAE'      : round(mae_val,  6),
        'MAPE'     : round(mape_val, 4),
    }

results_df = pd.DataFrame(results).T.sort_values('RMSE')

print("=" * 80)
print("  MODEL PERFORMANCE COMPARISON  (sorted by RMSE)")
print("=" * 80)
print(results_df.to_string())

best_model_name = results_df['RMSE'].idxmin()
print(f"\n{'─'*80}")
print(f"  ★ BEST MODEL: {best_model_name}  |  RMSE = {results_df.loc[best_model_name,'RMSE']:.6f}  |  MAPE = {results_df.loc[best_model_name,'MAPE']:.2f}%")
print(f"{'─'*80}")

# Quick visual: RMSE bar chart
fig, ax = plt.subplots(figsize=(10, 5))
colors = [PALETTE[i % len(PALETTE)] for i in range(len(results_df))]
bars = ax.barh(results_df.index, results_df['RMSE'], color=colors, edgecolor='white', height=0.55)
ax.set_xlabel('RMSE (lower is better)')
ax.set_title('Model Comparison -- RMSE', fontsize=14)
for bar, v in zip(bars, results_df['RMSE']):
    ax.text(v + 0.0002, bar.get_y() + bar.get_height()/2,
            f'{v:.5f}', va='center', fontsize=10)
bars[0].set_edgecolor('gold')
bars[0].set_linewidth(2.5)
plt.tight_layout()
plt.savefig('fig04_model_comparison.png', bbox_inches='tight')
plt.close()


# ---
# 
# ## 8. 🔮 4-Week Return Rate Forecast
# 
# Using the **best model** to predict return rates for the next 4 weeks.  
# The model outputs all 4 weeks in a single forward pass from the last 8 weeks of real data.
# 

# In[10]:


best_model = trained_models[best_model_name]

# Single forward pass
last_seq      = scaled[-WINDOW:].copy()
current_input = np.expand_dims(last_seq, axis=0)

pred_scaled  = best_model.predict(current_input, verbose=0)
future_preds = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

last_date    = weekly['Week'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=OUT_STEPS+1, freq='W-MON')[1:]

print("=" * 70)
print("  4-WEEK RETURN RATE FORECAST")
print("=" * 70)
print(f"  Model used     : {best_model_name}")
print(f"  Forecast period: {future_dates[0].date()}  to  {future_dates[-1].date()}")
print(f"{'─'*70}")

forecast_df = pd.DataFrame({
    'Week'                 : [f'Week+{i+1}' for i in range(OUT_STEPS)],
    'Date'                 : future_dates,
    'Predicted_Return_Rate': future_preds.round(4),
    'Predicted_%'          : (future_preds * 100).round(2),
})
print(forecast_df.to_string(index=False))

avg_forecast = future_preds.mean()
print(f"{'─'*70}")
print(f"  Average forecasted return rate: {avg_forecast:.4f}  ({avg_forecast*100:.2f}%)")
print(f"{'─'*70}")

# Forecast visualisation
HISTORY_TAIL = 20
hist_dates = weekly['Week'].iloc[-HISTORY_TAIL:]
hist_vals  = weekly['return_rate'].iloc[-HISTORY_TAIL:]

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(hist_dates, hist_vals, color='steelblue',
        linewidth=2, marker='o', markersize=4, label='Historical (last 20 weeks)')
ax.plot(future_dates, future_preds, color='crimson',
        linewidth=2.5, linestyle='--', marker='*',
        markersize=14, label=f'4-Week Forecast ({best_model_name})')
ax.axhline(weekly['return_rate'].mean(), color='red', linestyle=':',
           linewidth=1.5, alpha=0.6, label=f"Historical Mean = {weekly['return_rate'].mean():.3f}")
ax.axvspan(last_date, future_dates[-1], color='lightyellow', alpha=0.4, label='Forecasting Horizon')
ax.axvline(last_date, color='gray', linestyle=':', linewidth=1.5, label='Now')

for d, v, lbl in zip(future_dates, future_preds, [f'Week+{i+1}' for i in range(OUT_STEPS)]):
    ax.annotate(f'{lbl}\n{v*100:.2f}%', (d, v),
                textcoords='offset points', xytext=(0, 14),
                ha='center', fontsize=10, color='crimson', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2))

ax.set_title(f'4-Week Return Rate Forecast -- {best_model_name}', fontsize=15, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Return Rate')
ax.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('fig05_future_forecast.png', bbox_inches='tight')
plt.close()

print(f"\nForecasted return rate for the next 4 weeks: ~{avg_forecast*100:.2f}%")
print("Now let's see which products are driving these returns →")


# ---
# 
# ## 9. 📦 Top 5 Products Driving Returns
# 
# We've forecasted a return rate of approximately **{avg}%** for the next 4 weeks.
# 
# **Now the key business question: Which products are causing the most returns?**
# 
# We analyse the most recent 8-week window of actual data to identify the **Top 5 products**
# that constitute the highest share of total returns.
# 

# In[11]:


# ── Analysis window: last WINDOW weeks of actual data ─────────────
analysis_weeks = weekly['Week'].iloc[-WINDOW:].tolist()
print(f"Analysis window: {analysis_weeks[0].date()} to {analysis_weeks[-1].date()}  ({WINDOW} weeks)")

df_recent = df[df['Week'].isin(analysis_weeks)].copy()

total_returns_in_window      = df_recent['is_returned'].sum()
total_transactions_in_window = len(df_recent)
window_return_rate           = total_returns_in_window / total_transactions_in_window

print(f"Total transactions : {total_transactions_in_window:,}")
print(f"Total returns      : {total_returns_in_window:,}")
print(f"Actual return rate : {window_return_rate*100:.2f}%")
print(f"Forecasted rate    : {avg_forecast*100:.2f}%  (next 4 weeks)")

# ── Per-product breakdown ─────────────────────────────────────────
product_returns = (
    df_recent
    .groupby('StockCode')
    .agg(
        description  = ('Description', 'first'),
        total_txns   = ('Invoice', 'count'),
        return_count = ('is_returned', 'sum'),
        avg_price    = ('Price', 'mean'),
    )
)
product_returns['return_rate'] = (
    product_returns['return_count'] / product_returns['total_txns']
)
product_returns['pct_of_total_returns'] = (
    product_returns['return_count'] / total_returns_in_window * 100
)

product_returns = product_returns[product_returns['return_count'] > 0]

# ── Top 5 by return count ────────────────────────────────────────
top5 = product_returns.sort_values('return_count', ascending=False).head(5)

print(f"\n{'='*80}")
print(f"  TOP 5 PRODUCTS DRIVING RETURNS")
print(f"  (Analysis: last {WINDOW} weeks  |  Forecasted rate ≈ {avg_forecast*100:.2f}%)")
print(f"{'='*80}")

for rank, (code, row) in enumerate(top5.iterrows(), 1):
    desc = str(row['description'])[:50]
    print(f"\n  #{rank}  StockCode: {code}")
    print(f"      Product        : {desc}")
    print(f"      Returns        : {int(row['return_count']):,}  out of  {int(row['total_txns']):,}  transactions")
    print(f"      Return Rate    : {row['return_rate']*100:.1f}%")
    print(f"      Share of Total : {row['pct_of_total_returns']:.1f}% of all returns")
    print(f"      Avg Price      : £{row['avg_price']:.2f}")

cumulative_share = top5['pct_of_total_returns'].sum()
print(f"\n{'─'*80}")
print(f"  ⚠️  These 5 products account for {cumulative_share:.1f}% of ALL returns in the window.")
print(f"{'─'*80}")


# In[12]:


# ── Visualisation: Top 5 Products ─────────────────────────────────
labels = [f"{code}\n{str(top5.loc[code,'description'])[:22]}" for code in top5.index]
x_pos  = range(len(labels))

fig, ax1 = plt.subplots(figsize=(14, 7))

# Bars: return count
bar_colors = ['#E74C3C', '#E67E22', '#F1C40F', '#3498DB', '#9B59B6']
bars = ax1.bar(x_pos, top5['return_count'].values,
               color=bar_colors, edgecolor='white', linewidth=1.5,
               width=0.55, zorder=3)
ax1.set_xlabel('Product (StockCode)', fontsize=12)
ax1.set_ylabel('Return Count', fontsize=12, color='#333')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(labels, fontsize=9)

# Annotate with count + share %
for bar, cnt, share in zip(bars, top5['return_count'].values,
                            top5['pct_of_total_returns'].values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{int(cnt)}  ({share:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Line: product return rate on secondary axis
ax2 = ax1.twinx()
ax2.plot(x_pos, top5['return_rate'].values * 100,
         color='crimson', linewidth=2.5, marker='D', markersize=9,
         markerfacecolor='white', markeredgecolor='crimson',
         markeredgewidth=2, zorder=5, label='Product Return Rate (%)')
for i, rate in enumerate(top5['return_rate'].values):
    ax2.annotate(f'{rate*100:.1f}%', (i, rate*100),
                 textcoords='offset points', xytext=(0, 12),
                 ha='center', fontsize=10, color='crimson', fontweight='bold')
ax2.set_ylabel('Product Return Rate (%)', fontsize=12, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

ax1.set_title(f'Top 5 Products Driving Returns\n'
              f'Forecasted Return Rate ≈ {avg_forecast*100:.2f}%  |  '
              f'These 5 products = {cumulative_share:.1f}% of all returns',
              fontsize=13, fontweight='bold', pad=18)
ax1.grid(axis='y', alpha=0.3)
fig.legend(loc='upper right', bbox_to_anchor=(0.97, 0.92), fontsize=10)
plt.tight_layout()
plt.savefig('fig06_top5_return_products.png', bbox_inches='tight')
plt.close()


# ---\n## 10. Summary

# In[13]:


print("=" * 70)
print("  PROJECT SUMMARY")
print("=" * 70)

print(f"\n  Models Trained: {', '.join(trained_models.keys())}")
print(f"  Best Model    : {best_model_name}  (RMSE = {results_df.loc[best_model_name,'RMSE']:.6f})")

print(f"\n  4-Week Forecast:")
for i in range(OUT_STEPS):
    print(f"    Week+{i+1}  ({future_dates[i].date()})  →  Return Rate = {future_preds[i]*100:.2f}%")
print(f"    Average : {avg_forecast*100:.2f}%")

print(f"\n  Top 5 Products Causing Returns:")
for rank, (code, row) in enumerate(top5.iterrows(), 1):
    desc = str(row['description'])[:35]
    print(f"    #{rank}  {code:12s}  {desc:35s}  Returns: {int(row['return_count']):4d}  ({row['pct_of_total_returns']:.1f}%)")

print(f"\n  These 5 products = {cumulative_share:.1f}% of all returns")
print("=" * 70)
print("  Pipeline complete.")
print("=" * 70)

