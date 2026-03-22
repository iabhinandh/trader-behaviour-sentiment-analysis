# Trader Behaviour & Performance Analytics

## Project Overview

This project analyses how Bitcoin market sentiment (Fear & Greed) influences trader behaviour and performance on the Hyperliquid decentralised exchange. By combining a daily sentiment index with granular trade-level data, we identify patterns in how traders make decisions and perform during different market conditions.

---

## Key Findings

- Fear days produce highest average net PnL ($7,410) — counter to expectations
- Traders place 2x bigger position sizes during Fear vs Greed days
- High Risk traders make 2.7x more profit on Fear days than Low Risk traders
- Frequent traders make 3.7x more profit on Fear days than Infrequent traders
- Consistent short bias (60%) maintained across all sentiment conditions
- Win rate stays consistent (~84%) regardless of market sentiment
- Random Forest model achieves 96% accuracy predicting daily profitability
- Win rate is the strongest predictor (72% importance) — trader skill > market conditions

---

## Datasets

### 1. Bitcoin Market Sentiment (Fear/Greed Index)
- Source: Alternative.me Fear & Greed Index
- Period: February 2018 — May 2025
- Rows: 2,644 daily records
- Key columns: date, value (1-100), classification, sentiment_binary

### 2. Historical Trader Data (Hyperliquid DEX)
- Source: Hyperliquid Decentralised Exchange
- Raw rows: 211,224 trades
- After cleaning: 104,272 closed trades
- Key columns: account, coin, size_usd, side, direction, closed_pnl, fee

---

## Project Structure
```
trader-behaviour-sentiment-analysis/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_modelling.ipynb
│
├── charts/
│   ├── chart_01_sentiment_distribution.png
│   ├── chart_02_pnl_by_sentiment.png
│   └── ... (14 charts total)
│
├── app.py                 ← Streamlit dashboard
└── README.md
```

---

## Notebook Summary

### 01 — Data Preparation
- Loaded and cleaned both datasets
- Filtered 211,224 trades to 104,272 closed trades
- Merged on date column
- Engineered daily metrics: daily_pnl, net_pnl, win_rate, trade_count, avg_size_usd, long_ratio

### 02 — Exploratory Data Analysis
- 8 visualisations across sentiment conditions
- Analysed PnL, trade frequency, position size, win rate, long/short ratio
- Extended analysis using 5-class classification (Extreme Fear → Extreme Greed)

### 03 — Analysis & Segmentation
- Statistical tests: t-test and Mann-Whitney U test
- 3 trader segments: Risk, Frequency, Performance
- Part C strategy recommendations based on findings

### 04 — Modelling
- Random Forest classifier for profitability prediction
- Handled class imbalance using class weights
- 96% accuracy, 81% recall on losing days
- Feature importance analysis

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Raw trades | 211,224 |
| Clean closed trades | 104,272 |
| Trader-day combinations | 1,690 |
| Unique traders | 32 |
| Date range | Dec 2023 — May 2025 |
| Greed days | 843 (50%) |
| Fear days | 571 (34%) |
| Model accuracy | 96% |

---

## Strategy Recommendations

**Strategy 1 — Fear Day Aggressive Trading**
During Fear days, High Risk + Frequent traders should maximise position sizes and trade frequency. Data shows 2.7x—3.7x more profit vs conservative traders.

**Strategy 2 — Greed Day Position Holding**
During Greed days, Inconsistent/aggressive traders should widen profit targets. Data shows 2.4x more profit vs Consistent traders.

**Strategy 3 — Avoid Neutral Days**
All segments should reduce activity during Neutral days. Low Risk traders generate only $1,357 vs $4,266 on Fear days.

---

## Tech Stack
```
Python, Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (Random Forest, KMeans, StandardScaler)
Streamlit (Interactive Dashboard)
Jupyter Notebook
```

---

## Data Limitation Note

Win rates of 84-86% are unusually high compared to real-world benchmarks (40-65%). This may indicate curated top-performer data or survivorship bias. All findings should be interpreted with this in mind.

---

## Author

**Abhinandh**
Entry-level Data Analyst
[GitHub](https://github.com/iabhinandh) | [LinkedIn](https://linkedin.com/in/abhinandh-o-143016348)
