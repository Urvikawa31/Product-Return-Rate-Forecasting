import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as stats
import os

def ensure_diag_dir():
    if not os.path.exists('scratch/diagnostics'):
        os.makedirs('scratch/diagnostics')

def check_stationarity(series, name="Series"):
    """Performs ADF test and determines if differencing is needed."""
    print(f"\n--- Stationarity Check: {name} ---")
    result = adfuller(series)
    p_value = result[1]
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    is_stationary = p_value < 0.05
    if is_stationary:
        print("Conclusion: Series is Stationary.")
    else:
        print("Conclusion: Series is Non-Stationary. Transformation (differencing) suggested.")
    
    return is_stationary, p_value

def plot_identification(series, name="Series"):
    """Generates ACF and PACF plots."""
    ensure_diag_dir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    plot_acf(series, ax=axes[0], title=f"ACF: {name}")
    plot_pacf(series, ax=axes[1], title=f"PACF: {name}")
    
    plt.tight_layout()
    plot_path = f"scratch/diagnostics/{name.lower()}_identification.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Identification plots saved to {plot_path}")

def residual_analysis(residuals, model_name):
    """Performs QQ-plot and Ljung-Box test on residuals."""
    ensure_diag_dir()
    print(f"\n--- Residual Analysis: {model_name} ---")
    
    # Ljung-Box Test
    # If p-values are > 0.05, residuals are uncorrelated (white noise)
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    p_val = lb_test['lb_pvalue'].values[0]
    print(f"Ljung-Box p-value (lag 10): {p_val:.4f}")
    
    if p_val > 0.05:
        print("Conclusion: Residuals appear uncorrelated (Ideal).")
    else:
        print("Conclusion: Residuals show autocorrelation (Consider refining model).")
        
    # QQ Plot
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot: {model_name} Residuals")
    qq_path = f"scratch/diagnostics/{model_name.lower().replace(' ', '_')}_qq.png"
    plt.savefig(qq_path)
    plt.close()
    print(f"QQ plot saved to {qq_path}")
    
    return p_val
