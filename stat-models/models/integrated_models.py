import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm

def run_arima_forecast(train, test_len, order=(2, 1, 2)):
    """ARIMA model with static multi-step forecast."""
    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=test_len)
    return forecast

def run_rolling_arima_forecast(train, test, order=(2, 1, 2), bias_adjustment=True):
    """
    Improved ARIMA with Rolling Forecast, Log-Transformation, and Business-Bias Adjustment.
    """
    # 1. Log transform to stabilize variance
    # Add small constant to avoid log(0)
    history = list(np.log1p(train.values))
    predictions = []
    
    # Costs: Under=100, Over=50. Optimal quantile = 100/(100+50) = 0.667
    target_quantile = 100 / (100 + 50)
    z_score = norm.ppf(target_quantile) 
    
    for t in range(len(test)):
        # Fit model on log data
        model_fit = ARIMA(history, order=order).fit()
        
        # Get point forecast and forecast variance
        forecast_obj = model_fit.get_forecast(steps=1)
        yhat_log = forecast_obj.predicted_mean[0]
        yhat_std = np.sqrt(forecast_obj.var_pred_mean[0])
        
        # Apply Quantile Shift (Business-Bias) if requested
        if bias_adjustment:
            yhat_log_biased = yhat_log + (z_score * yhat_std)
        else:
            yhat_log_biased = yhat_log
            
        # Back-transform to original scale
        yhat = np.expm1(yhat_log_biased)
        
        predictions.append(yhat)
        
        # Update history with log of actual value
        history.append(np.log1p(test.iloc[t]))
        
    return pd.Series(predictions, index=test.index)

def run_sarima_forecast(train, test_len, order=(1, 1, 1), s_order=(1, 1, 1, 7)):
    """SARIMA model with dynamic seasonal order."""
    model = SARIMAX(train, order=order, seasonal_order=s_order).fit(disp=False)
    forecast = model.forecast(steps=test_len)
    return forecast
