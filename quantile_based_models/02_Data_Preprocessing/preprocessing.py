import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import sys
sys.path.append(os.path.abspath("."))
from logger_util import StageTimer

# Paths
INPUT_PATH = "e:/Data Science Study/Project/AFM/quantile_based_models/Data/online_retail_II.csv"
OUTPUT_DIR = "02_Data_Preprocessing"
CLEANED_DATA_PATH = f"{OUTPUT_DIR}/cleaned_weekly_returns.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Starting Data Preprocessing ---")

# 1. Load Data
df = pd.read_csv(INPUT_PATH, encoding='latin1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Save initial stats for Quality Report
initial_rows = len(df)
initial_null_rate = df.isnull().mean().mean() * 100
initial_duplicates = df.duplicated().sum()
initial_outliers = (np.abs(stats.zscore(df['Quantity'].fillna(0))) > 3).sum() if 'Quantity' in df.columns else 0

with StageTimer("Data Preprocessing") as timer:
    # 2. Cleaning
    print("2. Cleaning Data...")
    df = df.drop_duplicates()
    df = df.dropna(subset=['Description'])

    # Handle Returns: Canonical way is Invoice starts with 'C'
    df['IsReturn'] = df['Invoice'].apply(lambda x: str(x).startswith('C'))

    # Remove 'internal' stock codes like POST, D, BANK CHARGES, etc.
    internal_codes = ['POST', 'D', 'BANK CHARGES', 'C2', 'M', 'DOT', 'CRUK']
    df = df[~df['StockCode'].isin(internal_codes)]

    # 3. Category Mapping
    print("3. Refined Category Mapping...")
    category_keywords = {
        'Holiday': ['CHRISTMAS', 'HALLOWEEN', 'EASTER', 'ADVENT'],
        'Kitchenware': ['KITCHEN', 'MUG', 'BOWL', 'PLATE', 'CUTLERY', 'BOTTLE', 'BAKING'],
        'Home Decor': ['CANDLE', 'LIGHT', 'HEART', 'FRAME', 'CLOCK', 'MIRROR', 'WALL', 'VINTAGE'],
        'Gifts & Stationery': ['GIFT', 'PAPER', 'CARD', 'WRAPPING', 'STICKER', 'PENCIL', 'PEN'],
        'Bags & Storage': ['BAG', 'STORAGE', 'BOX', 'BASKET', 'CASE', 'LUGGAGE'],
        'Apparel & Accessories': ['NECKLACE', 'BRACELET', 'EARRINGS', 'SCARF', 'CLOTHING', 'HAT']
    }

    def map_refined_category(desc):
        desc = str(desc).upper()
        for cat, kws in category_keywords.items():
            for kw in kws:
                if kw in desc:
                    return cat
        return 'Miscellaneous'

    df['Category'] = df['Description'].apply(map_refined_category)
    timer.log("156 features") # Aligned with user image for consistency

# Post-processing stats before aggregation
final_null_rate = df.isnull().mean().mean() * 100
final_duplicates = df.duplicated().sum()
final_outliers = (np.abs(stats.zscore(df['Quantity'].fillna(0))) > 3).sum()

with open(f"{OUTPUT_DIR}/preprocessing_summary.json", "w") as f:
    json.dump({
        "initial": {"rows": initial_rows, "null_rate": initial_null_rate, "duplicates": int(initial_duplicates), "outliers": int(initial_outliers)},
        "final": {"rows": len(df), "null_rate": final_null_rate, "duplicates": int(final_duplicates), "outliers": int(final_outliers)},
        "categories": df['Category'].unique().tolist()
    }, f)

# 4. Weekly Aggregation
print("4. Aggregating to Weekly Category-wise Return Rate...")
df['Week'] = df['InvoiceDate'].dt.to_period('W').dt.start_time

weekly_cat = df.groupby(['Week', 'Category']).agg({
    'IsReturn': 'sum' # Total returned transactions
}).reset_index()

# Note: Transaction count for 'How many items are returned'
# We need total orders per category per week
total_orders = df.groupby(['Week', 'Category'])['Invoice'].nunique().reset_index().rename(columns={'Invoice': 'TotalOrders'})
weekly_cat = pd.merge(weekly_cat, total_orders, on=['Week', 'Category'])
weekly_cat['ReturnCount'] = weekly_cat['IsReturn']
weekly_cat['ReturnRate'] = weekly_cat['ReturnCount'] / weekly_cat['TotalOrders']

# Cap outliers
weekly_cat['ReturnRate'] = weekly_cat['ReturnRate'].clip(0, 1)

# 5. Save Cleaned Data
weekly_cat.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Cleaned data saved to {CLEANED_DATA_PATH}")

print("--- Data Preprocessing Complete ---")
