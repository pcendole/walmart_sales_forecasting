import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "artifacts",
    "walmart_histgbr_artifacts.joblib"
)

REQUIRED_COLUMNS = [
    "Store", "Dept", "Date", "IsHoliday",
    "Temperature", "Fuel_Price",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "CPI", "Unemployment", "Type", "Size"
]


def load_artifacts(path=ARTIFACT_PATH):
    print(f"Loading artifacts from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact file not found: {path}")
    artifacts = joblib.load(path)
    return artifacts


def validate_input_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )


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

    if "IsHoliday" in df.columns:
        df["WeeksToHoliday"] = compute_weeks_to_holiday(df["Date"], df["IsHoliday"])
    else:
        df["WeeksToHoliday"] = 0

    return df


def preprocess(df: pd.DataFrame, encoders, feature_cols):
    df = df.copy()
    validate_input_columns(df)

    df = add_date_features(df)

    markdown_cols = [c for c in df.columns if c.startswith("MarkDown")]
    for col in markdown_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in ["CPI", "Unemployment", "Temperature", "Fuel_Price", "Size"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["CPI", "Unemployment"]:
        if col in df.columns:
            df[col] = df.groupby("Store")[col].transform(lambda s: s.ffill().bfill())

    df["IsHoliday"] = df["IsHoliday"].astype(bool)

    for col, le in encoders.items():
        if col not in df.columns:
            raise ValueError(f"Required encoded column missing: {col}")
        try:
            df[col] = le.transform(df[col])
        except Exception as e:
            raise ValueError(
                f"Encoding failed for column '{col}'. "
                f"Check whether input contains unseen values. Original error: {e}"
            )

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(
            "After preprocessing, these required feature columns are missing: "
            + ", ".join(missing_features)
        )

    X = df[feature_cols].copy()
    return X, df


def predict_csv(input_path: str, output_path: str):
    print(f"Reading input: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    artifacts = load_artifacts()
    model = artifacts["model"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]

    print("Preprocessing input data...")
    X, df_processed = preprocess(df, encoders, feature_cols)

    print("Running predictions...")
    preds = model.predict(X)

    df_out = df_processed.copy()
    df_out["Weekly_Sales_Pred"] = preds

    if {"Store", "Dept", "Date"}.issubset(df_out.columns):
        df_out["Id"] = (
            df_out["Store"].astype(str) + "_"
            + df_out["Dept"].astype(str) + "_"
            + pd.to_datetime(df_out["Date"]).dt.strftime("%Y-%m-%d")
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    df_out.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")
    print("\nPreview:")
    print(df_out.head())


def main():
    warnings.filterwarnings("default")

    parser = argparse.ArgumentParser(
        description="Walmart weekly sales forecasting using pretrained HistGradientBoosting model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV."
    )
    parser.add_argument(
        "--output",
        default="predictions.csv",
        help="Path to output CSV with predictions."
    )
    args = parser.parse_args()

    predict_csv(args.input, args.output)


if __name__ == "__main__":
    main()