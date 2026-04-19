import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
sys.path.append(os.path.abspath("."))
from logger_util import StageTimer

# Create results directory if it doesn't exist
OUTPUT_DIR = "01_Data_Overview/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Starting Data Overview and Characteristics ---")

# Load Data
def load_data(path):
    print(f"Loading data from {path}...")
    # Use latin1 encoding as discovered during research
    df = pd.read_csv(path, encoding='latin1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

with StageTimer("Data Overview (Collection)") as timer:
    df_raw = load_data("e:/Data Science Study/Project/AFM/quantile_based_models/Data/online_retail_II.csv")
    timer.log(f"{len(df_raw):,} records")

# 1. Basic Data Exploration
print("1. Basic Exploration...")
summary_stats = df_raw.describe(include='all')
summary_stats.to_csv(f"{OUTPUT_DIR}/basic_summary_statistics.csv")

print(f"Total entries: {len(df_raw)}")
print(f"Columns: {df_raw.columns.tolist()}")

# 2. Statistical Data Exploration
print("2. Statistical Exploration...")
# Missing values
missing = df_raw.isnull().sum()
missing_pct = (df_raw.isnull().sum() / len(df_raw)) * 100
missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
missing_df.to_csv(f"{OUTPUT_DIR}/missing_values_report.csv")

# Identify Returns: Invoice starting with 'C' (cancel) or negative Quantity
df_raw['IsReturn'] = df_raw['Invoice'].apply(lambda x: str(x).startswith('C'))
df_raw['IsReturn'] = df_raw['IsReturn'] | (df_raw['Quantity'] < 0)

# Statistical distribution of Returns
return_stats = df_raw.groupby('IsReturn')['Quantity'].describe()
return_stats.to_csv(f"{OUTPUT_DIR}/return_vs_normal_stats.csv")

with StageTimer("Data Overview (Analysis)") as timer:
    # 3. Advanced EDA: Time Series Analysis
    print("3. Time Series Analysis...")
    # Aggregate to Weekly
    df_raw['Date'] = df_raw['InvoiceDate'].dt.date
    weekly_stats = df_raw.set_index('InvoiceDate').resample('W').agg({
        'Invoice': 'nunique',
        'Quantity': 'sum',
        'IsReturn': 'sum'
    }).rename(columns={'Invoice': 'OrderCount', 'IsReturn': 'ReturnCount'})

    weekly_stats['ReturnRate'] = weekly_stats['ReturnCount'] / weekly_stats['OrderCount']

    plt.figure(figsize=(12, 6))
    plt.plot(weekly_stats.index, weekly_stats['ReturnRate'], label='Weekly Return Rate', color='red')
    plt.title('Weekly Aggregate Product Return Rate')
    plt.xlabel('Date')
    plt.ylabel('Return Rate')
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/weekly_return_rate_trend.png")
    plt.close()
    timer.log(f"{len(weekly_stats)} patterns")

# 4. Business & Economic Exploration: Country-wise
print("4. Business Oriented Exploration...")
country_returns = df_raw.groupby('Country').agg({
    'Quantity': 'sum',
    'IsReturn': 'sum'
})
country_returns['ReturnRate'] = country_returns['IsReturn'] / country_returns['Quantity'].abs()
country_returns = country_returns.sort_values('ReturnRate', ascending=False)
country_returns.head(15).to_csv(f"{OUTPUT_DIR}/top_returning_countries.csv")

# 5. Advanced EDA: Category Identification (Simplified)
print("5. Category Wise Analysis...")
# Map top keywords to categories
category_map = {
    'CHRISTMAS': 'Holiday',
    'HEART': 'Valentine/Gifts',
    'BAG': 'Luggage/Storage',
    'BOTTLE': 'Kitchenware',
    'KITCHEN': 'Kitchenware',
    'FLOWER': 'Gifts',
    'CANDLE': 'Decor',
    'MUG': 'Kitchenware'
}

def get_category(desc):
    if pd.isna(desc): return 'Others'
    desc = str(desc).upper()
    for kw, cat in category_map.items():
        if kw in desc: return cat
    return 'Others'

df_raw['Category'] = df_raw['Description'].apply(get_category)
cat_returns = df_raw.groupby('Category').agg({
    'Quantity': 'sum',
    'IsReturn': 'sum'
})
cat_returns['ReturnRate'] = cat_returns['IsReturn'] / cat_returns['Quantity'].abs()
cat_returns.to_csv(f"{OUTPUT_DIR}/category_wise_return_rate.csv")

# Plot Category Wise
plt.figure(figsize=(10, 6))
cat_returns['ReturnRate'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Return Rate by Product Category')
plt.xlabel('Return Rate')
plt.savefig(f"{OUTPUT_DIR}/category_return_rate_bar.png")
plt.close()

# 6. Dataset Overview and Characteristics (Table 1)
print("6. Generating Table 1...")
total_transactions = len(df_raw)
unique_users = df_raw['Customer ID'].nunique()
product_categories = df_raw['Category'].nunique()
overall_return_rate = df_raw['IsReturn'].mean() * 100
data_period_months = (df_raw['InvoiceDate'].max() - df_raw['InvoiceDate'].min()).days / 30

table_1 = pd.DataFrame({
    'Metric': ['Total Transactions', 'Unique Users', 'Product Categories', 'Return Rate', 'Data Period'],
    'Value': [
        f"{total_transactions:,}",
        f"{unique_users:,}",
        f"{product_categories}",
        f"{overall_return_rate:.1f}%",
        f"{data_period_months:.1f} months"
    ],
    'Description': [
        'Complete purchase records with return outcomes',
        'Individual customers across analysis period',
        'Distinct product classifications',
        'Overall proportion of returned purchases',
        'Temporal coverage for trend analysis'
    ]
})
table_1.to_csv(f"{OUTPUT_DIR}/Table_1_Dataset_Overview.csv", index=False)

print("--- Data Overview Complete. Results saved in 01_Data_Overview/results ---")
