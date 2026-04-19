import pandas as pd
import numpy as np

def analyze_data():
    try:
        df = pd.read_csv('data/online_retail_II_UCI.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Identify returns
        df['IsReturn'] = df['Invoice'].str.startswith('C', na=False) | (df['Quantity'] < 0)
        
        # Calculate daily quantities
        daily_sales = df[~df['IsReturn']].groupby(df['InvoiceDate'].dt.date)['Quantity'].sum()
        daily_returns = df[df['IsReturn']]['Quantity'].abs().groupby(df[df['IsReturn']]['InvoiceDate'].dt.date).sum()
        
        combined = pd.DataFrame({'Sales': daily_sales, 'Returns': daily_returns}).fillna(0)
        # Handle division by zero
        combined['Total'] = combined['Sales'] + combined['Returns']
        combined['ReturnRate'] = np.where(combined['Total'] > 0, combined['Returns'] / combined['Total'], 0)
        combined.index = pd.to_datetime(combined.index)
        
        # Fill missing dates with 0
        all_dates = pd.date_range(start=combined.index.min(), end=combined.index.max())
        combined = combined.reindex(all_dates, fill_value=0)
        
        print(f"Time range: {combined.index.min()} to {combined.index.max()}")
        print(f"Total Days: {len(combined)}")
        print(f"Mean Return Rate: {combined['ReturnRate'].mean():.4f}")
        print(f"Max Return Rate: {combined['ReturnRate'].max():.4f}")
        
        # Save for further use
        combined.to_csv('scratch/daily_return_rate.csv')
        print("Saved daily_return_rate.csv to scratch/")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_data()
