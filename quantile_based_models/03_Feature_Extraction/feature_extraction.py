import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))
from logger_util import StageTimer

# Paths
INPUT_PATH = "02_Data_Preprocessing/cleaned_weekly_returns.csv"
OUTPUT_DIR = "03_Feature_Extraction"
OUTPUT_PATH = f"{OUTPUT_DIR}/features_return_rate.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Starting Advanced Feature Extraction ---")

# 1. Load Data
df = pd.read_csv(INPUT_PATH)
df['Week'] = pd.to_datetime(df['Week'])
df = df.sort_values(['Category', 'Week'])

with StageTimer("Feature Extraction") as timer:
    # 2. Advanced Momentum Features
    print("2. Generating Momentum Features...")
    df['ReturnRate_Diff_1'] = df.groupby('Category')['ReturnRate'].diff(1)
    df['ReturnRate_Accel'] = df.groupby('Category')['ReturnRate_Diff_1'].diff(1)
    
    # 3. Generating Lagged Features
    print("3. Generating Lagged Features...")
    lags = [1, 2, 4, 8, 12]
    for lag in lags:
        df[f'ReturnRate_Lag_{lag}'] = df.groupby('Category')['ReturnRate'].shift(lag)

    # 4. Rolling Statistics (Mean, Std, Max)
    print("4. Generating Rolling Statistics...")
    windows = [4, 8, 12]
    for window in windows:
        df[f'ReturnRate_RollMean_{window}'] = df.groupby('Category')['ReturnRate'].transform(lambda x: x.shift(1).rolling(window=window).mean())
        df[f'ReturnRate_RollStd_{window}'] = df.groupby('Category')['ReturnRate'].transform(lambda x: x.shift(1).rolling(window=window).std())
        df[f'ReturnRate_RollMax_{window}'] = df.groupby('Category')['ReturnRate'].transform(lambda x: x.shift(1).rolling(window=window).max())

    # 5. Cyclical Seasonality
    print("5. Generating Cyclical Features...")
    df['WeekOfYear'] = df['Week'].dt.isocalendar().week
    df['Week_Sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
    df['Week_Cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)

    # 6. Categorical Encoding
    print("6. Categorical Encoding...")
    df = pd.get_dummies(df, columns=['Category'], prefix='Cat')

    # Drop NaNs
    df_features = df.dropna()
    timer.log(f"{len(df_features.columns)} features")

# 7. Save Features
df_features.to_csv(OUTPUT_PATH, index=False)
print(f"Features saved to {OUTPUT_PATH}. Shape: {df_features.shape}")
print("--- Advanced Feature Extraction Complete ---")
