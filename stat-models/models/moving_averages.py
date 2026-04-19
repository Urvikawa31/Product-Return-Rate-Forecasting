import pandas as pd
import numpy as np

def run_sma_forecast(train, test_len, window=7):
    """Simple Moving Average (SMA) forecast."""
    sma_val = train.rolling(window).mean().iloc[-1]
    return pd.Series([sma_val] * test_len)

def run_wma_forecast(train, test_len, window=7):
    """Weighted Moving Average (WMA) forecast."""
    weights = np.arange(1, window + 1)
    wma_val = (train.iloc[-window:] * weights).sum() / weights.sum()
    return pd.Series([wma_val] * test_len)
