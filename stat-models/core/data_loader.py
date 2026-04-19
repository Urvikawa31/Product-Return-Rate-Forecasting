import pandas as pd

def load_and_split_data(file_path='scratch/daily_return_rate.csv', test_size=30):
    """Loads cleaned daily return rate and performs temporal split."""
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    series = df['ReturnRate']
    
    split_point = len(series) - test_size
    train, test = series.iloc[:split_point], series.iloc[split_point:]
    return train, test
