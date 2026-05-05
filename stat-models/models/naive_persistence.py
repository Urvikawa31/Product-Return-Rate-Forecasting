import pandas as pd
import numpy as np

def run_naive_forecast(train, test_len, s=7):
    """
    Seasonal Naive model: repeats the last observed seasonal cycle.
    Default s=7 for weekly patterns in daily data.
    """
    last_season = train.iloc[-s:].values
    # Repeat the last season to cover the test length
    forecast = np.tile(last_season, int(np.ceil(test_len / s)))[:test_len]
    
    forecast_series = pd.Series(
        forecast, 
        index=pd.RangeIndex(len(train), len(train) + test_len)
    )
    return forecast_series
