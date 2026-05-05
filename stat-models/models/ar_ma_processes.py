import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings

def run_ar_forecast(train, test, lags=2):
    """Autoregressive (AR) model with rolling window."""
    history = list(train)
    predictions = []
    for t in range(len(test)):
        model = AutoReg(history, lags=lags).fit()
        predictions.append(model.predict(start=len(history), end=len(history))[0])
        history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)

def run_ma_forecast(train, test, order=2):
    """Moving Average (MA) process (via ARIMA 0,0,q) with rolling window."""
    history = list(train)
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(len(test)):
            model = ARIMA(history, order=(0, 0, order)).fit()
            predictions.append(model.forecast(steps=1)[0])
            history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)

def run_arma_forecast(train, test, order=(2, 2)):
    """ARMA model with rolling window."""
    history = list(train)
    predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(len(test)):
            model = ARIMA(history, order=(order[0], 0, order[1])).fit()
            predictions.append(model.forecast(steps=1)[0])
            history.append(test.iloc[t])
    return pd.Series(predictions, index=test.index)
