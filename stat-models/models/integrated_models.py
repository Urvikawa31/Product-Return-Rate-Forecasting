import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_arima_forecast(train, test_len, order=(2, 1, 2)):
    """ARIMA model with dynamic order selection."""
    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=test_len)
    return forecast

def run_sarima_forecast(train, test_len, order=(1, 1, 1), s_order=(1, 1, 1, 7)):
    """SARIMA model with dynamic seasonal order."""
    model = SARIMAX(train, order=order, seasonal_order=s_order).fit(disp=False)
    forecast = model.forecast(steps=test_len)
    return forecast
