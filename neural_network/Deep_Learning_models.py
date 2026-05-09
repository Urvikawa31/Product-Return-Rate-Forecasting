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


# ## 9.5 Diebold-Mariano Test (Statistical Significance)
#
# Tests whether forecast accuracy differences are statistically significant.
# - **TCN vs LSTM**
# - **TCN vs MLP**
#
# H₀: Both models have equal predictive accuracy
# H₁: The models have different predictive accuracy

from scipy import stats as sp_stats

def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    e1, e2: forecast error arrays (actual - predicted)
    Returns: DM statistic, p-value
    Positive DM stat means model 2 is better (lower loss).
    """
    d = e1**2 - e2**2          # squared-error loss differential
    d_mean = np.mean(d)
    T = len(d)
    gamma0 = np.var(d, ddof=1) # HAC variance (lag-0)
    dm_stat = d_mean / np.sqrt(gamma0 / T)
    p_value = 2 * (1 - sp_stats.t.cdf(np.abs(dm_stat), df=T - 1))
    return dm_stat, p_value

# ── Compute errors (original scale) ─────────────────────────────
e_tcn  = y_test_inv.flatten() - predictions['TCN'].flatten()
e_lstm = y_test_inv.flatten() - predictions['LSTM'].flatten()
e_mlp  = y_test_inv.flatten() - predictions['MLP'].flatten()

# ── Run DM tests ─────────────────────────────────────────────────
dm_stat_1, p_val_1 = dm_test(e_tcn, e_lstm)
dm_stat_2, p_val_2 = dm_test(e_tcn, e_mlp)

def sig_label(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "n.s."))

def winner(dm, m1, m2):
    return m1 if dm < 0 else (m2 if dm > 0 else "Tie")

dm_results = pd.DataFrame({
    'Comparison'    : ['TCN vs LSTM', 'TCN vs MLP'],
    'DM Statistic'  : [round(dm_stat_1, 4), round(dm_stat_2, 4)],
    'p-value'       : [f"{p_val_1:.6f}", f"{p_val_2:.6f}"],
    'Significance'  : [sig_label(p_val_1), sig_label(p_val_2)],
    'Better Model'  : [winner(dm_stat_1, 'TCN', 'LSTM'),
                       winner(dm_stat_2, 'TCN', 'MLP')],
})

print("\n" + "=" * 85)
print("  DIEBOLD-MARIANO TEST  (Statistical Significance of Forecast Accuracy)")
print("=" * 85)
print(f"\n  H0: Both models have equal predictive accuracy")
print(f"  H1: Models have significantly different accuracy\n")
print(dm_results.to_string(index=False))
print(f"\n  Legend: *** p<0.01 | ** p<0.05 | * p<0.10 | n.s. = not significant")
print("=" * 85)




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

