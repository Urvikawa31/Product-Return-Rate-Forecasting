import pandas as pd
import numpy as np
import os

def calculate_business_impact(csv_path='scratch/benchmark_comparison.csv', output_path='scratch/business_impact_metrics.csv', test_days=30):
    """
    Calculates business impact metrics based on model performance compared to a baseline.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run evaluation first.")
        return

    # Load results
    df = pd.read_csv(csv_path)
    
    # Identify Baseline (using Seasonal Naive as the new standard)
    baseline_model = 'Seasonal Naive (s=7)'
    if baseline_model not in df['Model'].values:
        print(f"Error: '{baseline_model}' not found in results for comparison.")
        return
        
    baseline_omc = df[df['Model'] == baseline_model]['OMC (Cost)'].values[0]
    
    # Constants
    weeks = test_days / 7
    
    # Calculations
    # 1. Avoided Weekly Loss ($): Savings per week
    df['Avoided Weekly Loss ($)'] = (baseline_omc - df['OMC (Cost)']) / weeks
    
    # 2. Logistics Efficiency (%): Percentage reduction in mismatch cost
    df['Logistics Efficiency (%)'] = (1 - (df['OMC (Cost)'] / baseline_omc)) * 100
    
    # 3. Annualized ROI (%): Return on Investment (Benefit / Cost)
    # Benefit = Cost Avoided, Cost = Mismatch cost remaining
    # This represents the ROI of switching from Naive to this model.
    # To avoid division by zero if OMC is 0
    df['Annualized ROI (%)'] = df.apply(
        lambda x: ((baseline_omc - x['OMC (Cost)']) / x['OMC (Cost)'] * 100) if x['OMC (Cost)'] != 0 else 100.0, 
        axis=1
    )

    # Note: Annualizing the ROI (%) specifically usually involves time, 
    # but since Efficiency is already a rate, Savings/Cost is a point-in-time ROI 
    # that remains constant over the year if performance is stable.
    
    # Select and format columns for report
    report_df = df[['Model', 'Avoided Weekly Loss ($)', 'Logistics Efficiency (%)', 'Annualized ROI (%)']]
    
    # Sort by efficiency (highest reduction first)
    report_df = report_df.sort_values(by='Logistics Efficiency (%)', ascending=False)
    
    # Save to CSV
    report_df.to_csv(output_path, index=False)
    print(f"Business impact performance metrics saved to {output_path}")
    
    return report_df

if __name__ == "__main__":
    calculate_business_impact()
