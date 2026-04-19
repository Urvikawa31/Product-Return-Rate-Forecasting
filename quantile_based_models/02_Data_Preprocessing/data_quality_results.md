# Data Quality and Preprocessing Report

## Preprocessing Summary
- Initial Rows: 1067371
- Initial Nulls: 247389
- Final Weekly Rows: 728
- Categories Found: ['Holiday', 'Home Decor', 'Bags & Storage', 'Miscellaneous', 'Kitchenware', 'Gifts & Stationery', 'Apparel & Accessories']

## Improvement Analysis
- **Sparsity**: Raw data had fragmented transactions. Aggregated data provides a consistent time-series per category.
- **Noise Reduction**: Internal stock codes and null descriptions were removed.
- **Target Engineering**: Developed a stabilized 'ReturnRate' metric suitable for probabilistic forecasting.
