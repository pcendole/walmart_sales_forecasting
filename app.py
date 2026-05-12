import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")

ARTIFACT_PATH = os.path.join("artifacts", "walmart_histgbr_artifacts.joblib")

REQUIRED_COLUMNS = [
    "Store", "Dept", "Date", "IsHoliday",
    "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Type", "Size"
]

@st.cache_resource
def load_artifacts(path=ARTIFACT_PATH):
    return joblib.load(path)

def validate_input_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

def compute_weeks_to_holiday(date_series: pd.Series, holiday_mask: pd.Series) -> pd.Series:
    date_series = pd.to_datetime(date_series).dt.tz_localize(None)
    holiday_dates = pd.to_datetime(date_series[holiday_mask.astype(bool)]).drop_duplicates().sort_values()

    if len(holiday_dates) == 0:
        return pd.Series(np.zeros(len(date_series), dtype=int), index=date_series.index)

    date_vals = date_series.astype("int64").to_numpy() // 10**9
    holiday_vals = holiday_dates.astype("int64").to_numpy() // 10**9

    weeks = np.empty(len(date_vals), dtype=int)
    sec_per_week = 7 * 24 * 3600

    for i, x in enumerate(date_vals):
        min_diff_sec = np.min(np.abs(holiday_vals - x))
        weeks[i] = int(min_diff_sec // sec_per_week)

    return pd.Series(weeks, index=date_series.index)

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["WeeksToHoliday"] = compute_weeks_to_holiday(df["Date"], df["IsHoliday"])

    return df

def preprocess(df: pd.DataFrame, encoders, feature_cols):
    df = df.copy()
    validate_input_columns(df)

    df = add_date_features(df)

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["CPI", "Unemployment", "Temperature", "Fuel_Price", "Size"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CPI", "Unemployment"]:
        df[col] = df.groupby("Store")[col].transform(lambda s: s.ffill().bfill())

    df["IsHoliday"] = df["IsHoliday"].astype(bool)

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    X = df[feature_cols].copy()
    return X, df

def predict_df(raw_df: pd.DataFrame):
    artifacts = load_artifacts()
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]

    X, processed_df = preprocess(raw_df, encoders, feature_cols)
    preds = model.predict(X)

    out = processed_df.copy()
    out["Weekly_Sales_Pred"] = preds
    out["Id"] = (
        out["Store"].astype(str) + "_"
        + out["Dept"].astype(str) + "_"
        + pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    )
    return out

@st.cache_data
def convert_df_to_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

st.title("Walmart Weekly Sales Forecasting Dashboard")
st.write("Upload merged Walmart input CSV and generate weekly sales forecasts.")

with st.expander("Required input columns"):
    st.write(REQUIRED_COLUMNS)

uploaded_file = st.file_uploader("Upload input CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

        if st.button("Run Forecast"):
            with st.spinner("Generating forecasts..."):
                pred_df = predict_df(raw_df)

            st.subheader("Prediction Preview")
            st.dataframe(pred_df.head(20), use_container_width=True)

            st.subheader("Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows predicted", f"{len(pred_df):,}")
            c2.metric("Mean predicted sales", f"{pred_df['Weekly_Sales_Pred'].mean():,.2f}")
            c3.metric("Max predicted sales", f"{pred_df['Weekly_Sales_Pred'].max():,.2f}")

            st.subheader("Filter and visualize")
            store_options = sorted(pred_df["Store"].dropna().unique().tolist())
            selected_store = st.selectbox("Select Store", store_options)

            filtered_store = pred_df[pred_df["Store"] == selected_store]
            dept_options = sorted(filtered_store["Dept"].dropna().unique().tolist())
            selected_dept = st.selectbox("Select Dept", dept_options)

            chart_df = (
                filtered_store[filtered_store["Dept"] == selected_dept]
                .copy()
                .sort_values("Date")
            )

            st.write(f"Showing Store {selected_store}, Dept {selected_dept}")
            st.line_chart(
                chart_df.set_index("Date")["Weekly_Sales_Pred"]
            )

            st.subheader("Filtered forecast table")
            st.dataframe(
                chart_df[["Store", "Dept", "Date", "Weekly_Sales_Pred", "Id"]],
                use_container_width=True
            )

            csv_data = convert_df_to_csv(pred_df)
            st.download_button(
                label="Download full forecast CSV",
                data=csv_data,
                file_name="walmart_forecast_output.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")