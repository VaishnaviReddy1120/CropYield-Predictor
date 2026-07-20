# 🌾 Crop Yield Predictor — India

## Problem Statement
Indian farmers make crop decisions based on intuition — without knowing 
the expected yield in advance. This leads to poor planning, resource wastage 
and financial losses for millions of farmers across India.

## Solution
Built an ML model that predicts crop yield (Tonnes/Hectare) based on:
- Crop type and Season
- State and Area of cultivation
- Weather conditions (Temperature, Rainfall)
- Farming inputs (Fertilizer, Pesticide)

Deployed as a live Streamlit web app — enabling real-time yield predictions 
for any crop, state and season combination across India.

## Dataset
- 19,689 records | 55 crops | 1997–2020
- Dropped incomplete 2020 data during cleaning (only 37 records)
- Enhanced with NASA temperature data
- Target: Yield (Tonnes/Hectare)

## Tech Stack
Python | Pandas | NumPy | Scikit-learn | XGBoost | Matplotlib | Seaborn | Streamlit

## Key Steps
- **EDA** — 9 plots revealing Coconut as major outlier, 2020 data incomplete
- **Data Cleaning** — Dropped 2020 data, removed Production column (data leakage prevention)
- **Outlier Handling** — IQR-based Winsorization (3060 outliers capped at 5.08 T/Ha)
- **Feature Engineering** — Created Temperature Range (Max - Min Temperature)
- **Label Encoding** — Converted Crop, Season, State to numeric values

## Model Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.14 | 1.53 |
| Random Forest | **0.92** | **0.48** |
| XGBoost | 0.91 | 0.51 |

## Key Findings
- Crop type is strongest predictor (importance: 0.50)
- Season is second most important (importance: 0.18)
- Linear Regression failed — crop yield has complex non-linear relationships
- Random Forest achieved best R² of 0.92 with RMSE of 0.48 Tonnes/Hectare

## Streamlit App
Live web app where users select crop, state, season and weather conditions 
to get instant yield predictions with farming advice.

## How to Run
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit
streamlit run app.py
