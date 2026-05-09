# Product Return Rate Forecasting: A Multi-Model Comparative Study on E-Commerce Transaction Data

Link : [Product Return Rate Forecasting UI Dashboard][https://deft-kitsune-b2e70b.netlify.app]

---

# Product Return Rate Forecasting  
### A Multi-Model Comparative Study on E-Commerce Transaction Data

> A research-oriented comparative forecasting framework that evaluates **Statistical Models**, **Deterministic Deep Learning Architectures**, and **Probabilistic Quantile Networks** for forecasting e-commerce product return rates under asymmetric operational costs.

---

# Authors

| Name | Institute |
|------|------------|
| Urvi Kava | Dhirubhai Ambani University |
| Harsh Patel | Dhirubhai Ambani University |
| Mannan Agrawal | Dhirubhai Ambani University |

### Instructor
**Prof. Pritam Anand**  
Dhirubhai Ambani University

---

# Abstract

Product returns are among the most expensive operational challenges in modern e-commerce systems. Reverse logistics, refund processing, warehouse overload, and inventory disruptions create major financial risks for retailers.

This project presents a **business-cost-aware forecasting framework** for predicting weekly product return rates using the **UCI Online Retail II Dataset** containing over **1 million transactions** collected over **24.6 months**.

The study compares three major forecasting paradigms:

- Classical Statistical Forecasting
- Deterministic Deep Learning Models
- Probabilistic Quantile Forecasting Networks

Unlike traditional forecasting systems that focus only on prediction accuracy, this work introduces **cost-aware operational evaluation metrics** such as:

- Operational Mismatch Cost (OMC)
- Order Management Cost (OMgC)
- Prediction Interval Coverage Probability (PICP)
- Continuous Ranked Probability Score (CRPS)
- Business ROI-based evaluation

The research demonstrates that:

- **SARIMA** performs best among statistical models.
- **Temporal Convolutional Networks (TCN)** dominate deterministic forecasting.
- **Conformal Quantile TCN (Q-TCN)** achieves the best probabilistic forecasting performance with:
  - **PICP = 1.00**
  - **Pinball Loss = 0.0142**
  - **Annualized ROI = +1371%**

The study concludes that calibrated probabilistic forecasting is not only statistically superior but also significantly more profitable in operational environments with asymmetric business costs.

---

# Table of Contents

- [Project Motivation](#project-motivation)
- [Research Objectives](#research-objectives)
- [Dataset Information](#dataset-information)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Business Impact](#business-impact)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [Research Contributions](#research-contributions)
- [References](#references)

---

# Project Motivation

E-commerce return management is extremely challenging because:

- Return rates are highly volatile
- Return signals are sparse
- Operational costs are asymmetric
- Reverse logistics are expensive
- Warehouse capacity planning depends on accurate forecasting

Under-forecasting causes:

- Warehouse congestion
- SLA violations
- Delayed refund processing

Over-forecasting causes:

- Resource wastage
- Excess staffing costs
- Inefficient reserve allocation

Traditional forecasting methods optimize symmetric errors such as RMSE or MAE, but real-world operational systems require **cost-sensitive forecasting**.

This project addresses this gap through:
- Cost-aware forecasting
- Uncertainty estimation
- Conformal calibration
- Operational ROI analysis

---

# Research Objectives

The main objectives of this study are:

- Construct a clean weekly return-rate time series
- Compare statistical, deep learning, and probabilistic forecasting paradigms
- Design operationally meaningful evaluation metrics
- Apply conformal prediction for calibrated uncertainty estimation
- Translate forecasts into actionable business decisions
- Identify future research gaps in cost-aware forecasting

---

# Dataset Information

## Dataset Source

**UCI Machine Learning Repository**  
Dataset: **Online Retail II**

---

## Dataset Statistics

| Metric | Value |
|--------|--------|
| Total Transactions | 1,067,371 |
| Unique Customers | 5,942 |
| Unique Products | 4,631 |
| Product Categories | 7 |
| Observation Period | 24.6 Months |
| Weekly Observations | 106 |
| Return Transactions | 18,744 |
| Overall Return Rate | 2.20% |

---

## Features

| Column | Description |
|--------|-------------|
| Invoice | Transaction ID |
| StockCode | Product Identifier |
| Description | Product Name |
| Quantity | Units Sold/Returned |
| InvoiceDate | Transaction Timestamp |
| Price | Product Price |
| Customer ID | Customer Identifier |
| Country | Customer Country |

---

# Data Preprocessing

The preprocessing pipeline includes:

## Cleaning Operations

- Removed missing descriptions
- Removed duplicate records
- Removed administrative pseudo-products:
  - POST
  - BANK CHARGES
  - DISCOUNT
  - MANUAL
  - DOT

---

## Return Identification

Returns are identified using:

```python
Invoice.startswith("C")
```

---

## Weekly Aggregation

Weekly return rate:

```math
y_t = \frac{\sum IsReturn_i}{|T_t|}
```

---

## Feature Engineering

### Lag Features
- 1 week
- 2 weeks
- 4 weeks
- 8 weeks
- 12 weeks

### Rolling Statistics
- Rolling Mean
- Rolling Std
- Rolling Max

### Temporal Features
- Weekly seasonality encoding
- Sin/Cos cyclical encoding

### Category Features
- One-hot category vectors

---

# Project Architecture

```text
Raw Transaction Data
        ↓
Data Cleaning & Aggregation
        ↓
Feature Engineering
        ↓
Forecasting Pipelines
 ┌──────────────┬──────────────┬──────────────┐
 │ Statistical  │ Deep Models  │ Probabilistic│
 └──────────────┴──────────────┴──────────────┘
        ↓
Model Evaluation
        ↓
Business Cost Analysis
        ↓
Operational Decision Support
```

---

# Methodology

The project compares three forecasting paradigms.

---

# 1. Statistical Forecasting Models

Classical forecasting methods used:

- Seasonal Naive
- SMA
- WMA
- AR
- MA
- ARMA
- ARIMA
- SARIMA

---

## Best Statistical Model

### SARIMA(2,0,0)(0,1,1,7)

### Why SARIMA Works Best

- Captures weak weekly seasonality
- Handles stationary return-rate series
- Minimizes asymmetric operational cost

---

# 2. Deterministic Deep Learning Models

Deep learning architectures evaluated:

| Model | Description |
|------|-------------|
| MLP | Dense Feedforward Network |
| RNN | Vanilla Recurrent Network |
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| TCN | Temporal Convolutional Network |
| Ensemble | Combined Meta-Learner |

---

## Why TCN Dominates

TCN achieved the best forecasting performance because:

- Dilated causal convolutions
- Parallel sequence processing
- Stable gradients
- Large receptive field
- Better small-data generalization

---

# 3. Probabilistic Quantile Forecasting

Quantile forecasting models:

- Q-MLP
- Q-RNN
- Q-LSTM
- Q-GRU
- Q-TCN

---

## Probabilistic Framework

The project uses:

- Quantile Regression
- Weighted Pinball Loss
- Quantile Crossing Penalty
- Split Conformal Calibration

---

## Quantiles Predicted

```text
0.05 Quantile
0.50 Quantile
0.95 Quantile
```

---

# Evaluation Metrics

## Standard Metrics

- RMSE
- MAE
- MAPE
- NMSE

---

## Cost-Aware Metrics

### Operational Mismatch Cost (OMC)

Used for statistical forecasting.

```math
OMC = c_u \sum max(0, y_t - \hat{y}_t) + c_o \sum max(0, \hat{y}_t - y_t)
```

---

### Order Management Cost (OMgC)

Used for deep learning models.

```math
OMgC = |y_t - \hat{y}_t| \times N_{txn} \times p
```

---

## Probabilistic Metrics

- Pinball Loss
- PICP
- CRPS
- Interval Width
- Winkler Score
- ECE

---

# Results

---

# Statistical Models — Rolling 30-Day Test (Sorted by OMC)

| Model | RMSE | MAE | MAPE% | NMSE | OMC |
|---|---|---|---|---|---|
| **SARIMA** | **0.0799** | 0.0368 | **82.282** | **0.943** | **81.08** |
| Seasonal Naive *(s = 7)* | 0.0828 | **0.0354** | **79.117** | 1.013 | 87.52 |
| ARMA(2,2) | 0.0822 | 0.0406 | 90.693 | 0.998 | 90.98 |
| MA(2) | 0.0822 | 0.0406 | 90.759 | 0.998 | 91.04 |
| AR(2) | 0.0822 | 0.0406 | 90.715 | 0.998 | 91.10 |
| SMA-7 | 0.0844 | 0.0372 | 83.257 | 1.051 | 93.90 |
| Weighted MA | 0.0847 | 0.0391 | 87.356 | 1.059 | 98.18 |
| ARIMA (rolling) | 0.0880 | 0.0603 | 134.795 | 1.143 | 112.52 |

---

# Deep Learning Models — Neural Network Track

| Model | MSE | NMSE | RMSE | MAPE% | OMgC |
|---|---|---|---|---|---|
| **TCN** | **5.1e-5** | **1.124** | **0.0071** | **27.89** | **171.42** |
| LSTM | 6.2e-5 | 1.387 | 0.0079 | 29.71 | 179.90 |
| MLP | 1.0e-4 | 2.222 | 0.0100 | 37.22 | 225.29 |
| GRU | 1.9e-4 | 4.139 | 0.0136 | 53.28 | 345.40 |
| Ensemble | 3.0e-4 | 6.603 | 0.0172 | 64.16 | 406.70 |
| RNN | 3.2e-4 | 7.133 | 0.0179 | 69.84 | 428.67 |

---

# Probabilistic Models — Conformally Calibrated

| Model | Pinball | PICP | CRPS | Width | Winkler | ECE |
|---|---|---|---|---|---|---|
| **TCN** | **0.0142** | **1.000** | **0.0142** | **1.246** | **0.310** | **0.100** |
| LSTM | 0.0254 | 0.939 | 0.0254 | 1.490 | 0.385 | 0.039 |
| GRU | 0.0645 | 0.954 | 0.0645 | 3.105 | 0.778 | 0.054 |
| MLP | 0.0439 | 0.754 | 0.0439 | 1.934 | 0.655 | 0.146 |
| RNN | 0.0989 | 0.862 | 0.0989 | 4.436 | 1.185 | 0.039 |

---

# Key Findings

## Major Observations

### 1. TCN Dominates Across Paradigms
TCN achieved the best performance in:
- Deterministic forecasting
- Probabilistic forecasting
- Business ROI evaluation

---

### 2. Probabilistic Forecasting Is More Valuable

Point forecasts provide only a single estimate.

Probabilistic forecasting provides:
- Risk bounds
- Confidence intervals
- Operational flexibility
- Better business planning

---

### 3. Symmetric Error Metrics Are Misleading

Lower MAPE does NOT guarantee lower business cost.

Operational metrics are more important than traditional accuracy metrics in asymmetric environments.

---

# Business Impact

The forecasting system enables:

## Reverse Logistics Planning
- Warehouse staffing optimization
- Dock scheduling
- Return capacity estimation

---

## Refund Reserve Planning
- Better financial allocation
- Reduced reserve overestimation

---

## SKU-Level Triage
Identification of high-return products for:
- Quality audits
- Listing review
- Operational intervention

---

## Estimated Savings

### Q-TCN Forecasting System

| Metric | Value |
|------|------|
| Weekly Avoided Loss | £282.89 |
| Annualized ROI | +1371% |

---

# Tech Stack

## Programming Languages
- Python

## Libraries
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
- TensorFlow / Keras
- PyTorch
- Matplotlib
- Seaborn

## Forecasting Techniques
- ARIMA/SARIMA
- Deep Learning
- Quantile Regression
- Conformal Prediction

---

# Project Structure

```text
Product-Return-Rate-Forecasting/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Statistical_Models.ipynb
│   ├── Deep_Learning.ipynb
│   ├── Probabilistic_Models.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── statistical_models/
│   ├── deep_models/
│   ├── quantile_models/
│   ├── evaluation/
│
├── results/
│   ├── plots/
│   ├── metrics/
│
├── report/
│   ├── paper.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Urvikawa31/Product-Return-Rate-Forecasting
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

## Run Statistical Models

```bash
python src/statistical_models/train_sarima.py
```

---

## Run Deep Learning Models

```bash
python src/deep_models/train_tcn.py
```

---

## Run Probabilistic Models

```bash
python src/quantile_models/train_qtcn.py
```

---

# Future Work

Future research directions include:

- Cost-aware quantile optimization
- Hybrid SARIMA-TCN models
- Real-time online conformal calibration
- Transformer-based forecasting
- Hierarchical forecasting reconciliation
- Exogenous signal integration:
  - Weather
  - Promotions
  - Holidays
  - Macroeconomic indicators

---

# Research Contributions

This work contributes:

- A unified comparative forecasting benchmark
- Business-cost-aware evaluation methodology
- Weighted quantile forecasting framework
- Conformal calibration pipeline
- Operational ROI translation framework
- Forecast-driven decision support system

---

# References

1. Hyndman & Athanasopoulos — *Forecasting: Principles and Practice*
2. Box & Jenkins — *Time Series Analysis*
3. Hochreiter & Schmidhuber — *LSTM Networks*
4. Bai et al. — *Temporal Convolutional Networks*
5. UCI Machine Learning Repository

---

# Acknowledgement

We sincerely thank **Prof. Pritam Anand** for his guidance throughout the Applied Forecasting Algorithms course at Dhirubhai Ambani University.

We also acknowledge the UCI Machine Learning Repository for providing the Online Retail II dataset.

---

# License

This project is intended for academic and research purposes.

---

⭐ If you found this project useful, consider giving it a star.
