import pandas as pd

def run_naive_forecast(train, test_len):
    """Simple persistence model: uses last observed value."""
    forecast = pd.Series([train.iloc[-1]] * test_len, index=pd.RangeIndex(len(train), len(train)+test_len))
    return forecast
