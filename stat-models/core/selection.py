import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

def find_best_arima(train, max_p=5, max_q=5, d=None):
    """Grid search for best ARIMA parameters using AIC."""
    print("\n--- Model Selection: ARIMA ---")
    if d is None:
        # We assume stationarity was checked previously
        d = 0
    
    best_aic = np.inf
    best_order = (0, d, 0)
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(train, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
            except:
                continue
    
    print(f"Best ARIMA Order: {best_order} (AIC: {best_aic:.2f})")
    return best_order

def find_best_sarima(train, max_p=2, max_q=2, d=1, D=1, s=7):
    """Grid search for best Seasonal ARIMA parameters using AIC."""
    print("\n--- Model Selection: SARIMA ---")
    best_aic = np.inf
    best_params = None
    
    # Simple grid for demonstration (limited to 1,d,1 for speed)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for P in range(2): # Seasonal AR
                for Q in range(2): # Seasonal MA
                    try:
                        # Fixed the other parameters for speed in this context
                        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=False)
                        if model.aic < best_aic:
                            best_aic = model.aic
                            best_params = ((p, d, q), (P, D, Q, s))
                    except:
                        continue
                        
    print(f"Best SARIMA Params: {best_params} (AIC: {best_aic:.2f})")
    return best_params
