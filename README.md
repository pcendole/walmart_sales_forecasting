# Walmart Weekly Sales Forecasting System

Forecast weekly Walmart sales at the **store-department level** using historical sales, holidays, markdowns, store metadata, and macroeconomic variables.

## Project summary

This project builds an end-to-end retail forecasting workflow:

- Time-series analysis on aggregate, store-level, and store-department data
- Feature engineering from dates, holidays, markdowns, and external indicators
- A global machine learning model using `HistGradientBoostingRegressor`
- A reusable prediction pipeline in Python
- A Streamlit dashboard for CSV upload, forecasting, visualization, and download

## Business objective

Retail demand changes across stores, departments, holidays, and seasons.  
The objective is to predict weekly sales accurately enough to support:

- Inventory planning
- Holiday demand preparation
- Promotion timing
- Store-level operational decisions

## Dataset

Files used:
- `train.csv`
- `test.csv`
- `features.csv`
- `stores.csv`

Core variables:
- Store, Dept, Type, Size
- Weekly sales
- Holiday flag
- Temperature, Fuel_Price
- MarkDown1–MarkDown5
- CPI, Unemployment

Engineered variables:
- Year, Month, WeekOfYear, Day
- IsMonthStart, IsMonthEnd
- WeeksToHoliday

## Modeling workflow

### 1. Time-series analysis
Performed at:
- Total Walmart weekly sales
- Individual store level
- Store-department level

Methods used:
- Trend and seasonal plotting
- Rolling mean / rolling standard deviation
- ADF stationarity tests
- STL decomposition
- SARIMA baselines

### 2. Global machine learning model
Final forecasting model:
- `HistGradientBoostingRegressor`

Why this model:
- Fast on large tabular data
- Strong nonlinear learning
- Better practicality than slower Random Forest baseline

## Validation performance

Time-based split:
- Train: 2010–2011
- Validation: 2012

Results:
- **MAE:** 4,233.69
- **RMSE:** 6,863.12
- **R²:** 0.9037

## Most important features

Top drivers from permutation importance:
- Dept
- Size
- Store
- CPI
- Type
- Unemployment
- Temperature
- Month
- WeekOfYear
- WeeksToHoliday

## Repository structure

```text
walmart_sales_forecasting/
├── artifacts/
│   └── walmart_histgbr_artifacts.joblib
├── data/
│   ├── input_test.csv
│   └── pred_output.csv
├── src/
│   └── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run CLI forecasting

```powershell
python -m src.predict --input data\input_test.csv --output data\pred_output.csv
```

## Run Streamlit dashboard

```powershell
streamlit run app.py
```

## Expected input columns

The app and CLI expect a merged CSV with:

- Store
- Dept
- Date
- IsHoliday
- Temperature
- Fuel_Price
- MarkDown1
- MarkDown2
- MarkDown3
- MarkDown4
- MarkDown5
- CPI
- Unemployment
- Type
- Size

## Key outcomes

- Built a reusable forecasting pipeline from experimentation to deployment-style inference
- Combined time-series diagnostics with a panel ML forecasting model
- Saved model artifacts for local prediction and dashboard use
- Created both command-line and dashboard interfaces for forecast generation

## Future work

- Add lag-based sales features
- Compare XGBoost / LightGBM
- Add explainability per store-department
- Deploy Streamlit online
- Add retraining pipeline

## Author

**Prathamesh Endole**  
Computer Engineering Student, Pune