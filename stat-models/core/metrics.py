import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

UNDERESTIMATION_COST = 100
OVERESTIMATION_COST = 50

def calculate_omc(y_true, y_pred):
    """Business Impact Metric: Operational Mismatch Cost"""
    diff = y_true - y_pred
    under_cost = np.sum(np.maximum(0, diff) * UNDERESTIMATION_COST)
    over_cost = np.sum(np.maximum(0, -diff) * OVERESTIMATION_COST)
    return under_cost + over_cost

def calculate_mape(y_true, y_pred):
    """
    Weighted Mean Absolute Percentage Error (WMAPE/WAPE).
    More robust for return rate forecasting where actuals can be zero or very small.
    Formula: (sum|y_true - y_pred| / sum|y_true|) * 100
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else np.inf
    return (np.sum(np.abs(y_true - y_pred)) / sum_actual) * 100

def calculate_nmse(y_true, y_pred):
    """Normalized Mean Squared Error (normalized by variance of actuals)"""
    mse = mean_squared_error(y_true, y_pred)
    var_actual = np.var(y_true)
    return mse / var_actual if var_actual != 0 else mse

def get_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    nmse = calculate_nmse(y_true, y_pred)
    omc = calculate_omc(y_true, y_pred)
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'NMSE': nmse,
        'OMC (Cost)': omc
    }
