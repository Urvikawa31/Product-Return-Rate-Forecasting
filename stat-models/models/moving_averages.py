import pandas as pd
import numpy as np

def run_sma_forecast(train, test, window=7):
    """Simple Moving Average (SMA) forecast with rolling window."""
    history = list(train)
    predictions = []
    for t in range(len(test)):
        sma_val = np.mean(history[-window:])
        predictions.append(sma_val)
        history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)

def run_wma_forecast(train, test, window=7):
    """Weighted Moving Average (WMA) forecast with rolling window."""
    history = list(train)
    predictions = []
    weights = np.arange(1, window + 1)
    weight_sum = weights.sum()
    for t in range(len(test)):
        wma_val = np.sum(np.array(history[-window:]) * weights) / weight_sum
        predictions.append(wma_val)
        history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)

