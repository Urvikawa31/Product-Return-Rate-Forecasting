import json
import pandas as pd
import numpy as np
import os
from core.metrics import get_metrics
from generate_business_metrics import calculate_business_impact


def main():
    print("Evaluating Advanced Benchmarks...")
    
    # Load data
    try:
        with open('scratch/all_forecasts.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: Experiment results not found. Please run 'python run_experiment.py' first.")
        return

    actual = np.array(data['Actual'])
    results = []
    
    # Evaluate each model
    for model_name, forecast in data.items():
        if model_name == 'Actual':
            continue
        
        metrics = get_metrics(actual, np.array(forecast), model_name)
        results.append(metrics)
    
    # Create comparison table
    results_df = pd.DataFrame(results)
    
    # Sort by OMC (Business impact)
    results_df = results_df.sort_values(by='OMC (Cost)')
    
    print("\n" + "="*80)
    print("PRODUCT RETURN RATE FORECASTING: ADVANCED BENCHMARK EVALUATION")
    print("="*80)
    # Customize display format for clarity
    pd.options.display.float_format = '{:,.4f}'.format
    print(results_df.to_string(index=False))
    print("="*80)
    
    # 1. Save to CSV (Single file as requested)
    csv_path = 'scratch/benchmark_comparison.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nBenchmark table saved to {csv_path}")
    
    # 2. Save to TXT with formatting and title
    best_model = results_df.iloc[0]['Model']
    txt_path = 'scratch/benchmark_evaluation_comparison.txt'
    with open(txt_path, 'w') as f:
        f.write("="*90 + "\n")
        f.write("PRODUCT RETURN RATE FORECASTING: MODEL BENCHMARK COMPARISON\n")
        f.write("="*90 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n" + "="*90 + "\n")
        f.write(f"RECOMMENDED MODEL based on OMC: {best_model}\n")
        f.write("="*90 + "\n")
    
    print(f"Formatted report saved to {txt_path}")
    
    # --- VISUALIZATION: Actual vs Top 3 Models ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 7))
    plt.plot(actual[-50:], label='Actual', color='black', linewidth=2, marker='o') # Last 50 days for clarity
    
    # Plot top 3 models
    top_models = results_df.head(3)['Model'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, model_name in enumerate(top_models):
        forecast = np.array(data[model_name])
        plt.plot(forecast[-50:], label=model_name, linestyle='--', color=colors[i])
        
    plt.title("Product Return Rate Forecasting: Top 3 Models vs Actual (Last 50 Days)")
    plt.xlabel("Days in Test Set")
    plt.ylabel("Return Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if not os.path.exists('scratch/plots'):
        os.makedirs('scratch/plots')
    
    plot_path = 'scratch/plots/forecast_comparison_top3.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparison plot saved to {plot_path}")
    
    # --- VISUALIZATION: Error Distribution ---
    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(top_models):
        error = actual - np.array(data[model_name])
        plt.hist(error, bins=30, alpha=0.5, label=model_name)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Forecast Error Distribution")
    plt.legend()
    hist_path = 'scratch/plots/error_distribution.png'
    plt.savefig(hist_path)
    plt.close()
    print(f"Error distribution plot saved to {hist_path}")

    # --- BUSINESS IMPACT METRICS ---
    calculate_business_impact()

    print(f"RECOMMENDED MODEL: {best_model}")

if __name__ == "__main__":
    main()
