import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

def run_ar_forecast(train, test_len, lags=2):
    """Autoregressive (AR) model."""
    model = AutoReg(train, lags=lags).fit()
    forecast = model.predict(start=len(train), end=len(train)+test_len-1)
    return pd.Series(forecast.values)

def run_ma_forecast(train, test_len, order=2):
    """Moving Average (MA) process (via ARIMA 0,0,q)."""
    model = ARIMA(train, order=(0, 0, order)).fit()
    forecast = model.forecast(steps=test_len)
    return pd.Series(forecast.values)

def run_arma_forecast(train, test_len, order=(2, 2)):
    """ARMA model."""
    model = ARIMA(train, order=(order[0], 0, order[1])).fit()
    forecast = model.forecast(steps=test_len)
    return pd.Series(forecast.values)
