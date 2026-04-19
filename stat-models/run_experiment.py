import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import json
from core.data_loader import load_and_split_data
from models.naive_persistence import run_naive_forecast
from models.moving_averages import run_sma_forecast, run_wma_forecast
from models.ar_ma_processes import run_ar_forecast, run_ma_forecast, run_arma_forecast
from models.integrated_models import run_arima_forecast, run_sarima_forecast
from core.diagnostics import check_stationarity, plot_identification, residual_analysis
from core.selection import find_best_arima, find_best_sarima
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    print("Initializing Advanced Experiment with Diagnostics...")
    
    # 1. Load Data
    train, test = load_and_split_data()
    test_len = len(test)
    
    # 2. Stationarity and Identification (Workflow Step 1 & 2)
    is_stationary, p_val = check_stationarity(train, "Raw Daily Return Rate")
    d = 0
    work_series = train
    
    if not is_stationary:
        print("Applying First-Order Differencing (d=1)...")
        d = 1
        work_series = train.diff().dropna()
        check_stationarity(work_series, "Differenced Series")
    
    # Plot ACF/PACF for identification
    plot_identification(work_series, "Stationary_Series")
    
    # 3. Model Selection via AIC (Workflow Step 3)
    best_arima_order = find_best_arima(train, max_p=3, max_q=3, d=d)
    best_sarima_params = find_best_sarima(train, max_p=2, max_q=2, d=d, D=1, s=7)
    
    # 4. Running Models
    forecasts = {
        'Actual': test.values.tolist(),
        'Naive Baseline': run_naive_forecast(train, test_len).tolist(),
        'Moving Average (SMA-7)': run_sma_forecast(train, test_len).tolist(),
        'Weighted MA': run_wma_forecast(train, test_len).tolist(),
        'AR(2)': run_ar_forecast(train, test_len).tolist(),
        'MA(2)': run_ma_forecast(train, test_len).tolist(),
        'ARMA(2,2)': run_arma_forecast(train, test_len).tolist(),
        'ARIMA_Optimal': run_arima_forecast(train, test_len, order=best_arima_order).tolist(),
        'SARIMA_Optimal': run_sarima_forecast(train, test_len, order=best_sarima_params[0], s_order=best_sarima_params[1]).tolist()
    }
    
    # 5. Residual Analysis on Best Model (Workflow Step 4)
    # Re-fit the best SARIMA to get residuals
    print("\nPerforming Residual Analysis on SARIMA_Optimal...")
    final_model = SARIMAX(train, order=best_sarima_params[0], seasonal_order=best_sarima_params[1]).fit(disp=False)
    residuals = final_model.resid
    residual_analysis(residuals, "SARIMA_Optimal")
    
    # 6. Save results
    if not os.path.exists('scratch'):
        os.makedirs('scratch')
        
    with open('scratch/all_forecasts.json', 'w') as f:
        json.dump(forecasts, f)
        
    print("\nExperiment Complete. Forecasts saved to scratch/all_forecasts.json")
    print("Diagnostics saved to scratch/diagnostics/")

if __name__ == "__main__":
    main()
