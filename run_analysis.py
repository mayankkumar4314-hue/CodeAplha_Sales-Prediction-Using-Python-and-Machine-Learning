"""Run data cleaning, EDA, model comparison, and chart generation."""

import pandas as pd
import joblib

from sales_prediction import (
    DATA_PATH, OUTPUT_DIR, create_eda_charts, linear_coefficients, load_data,
    save_evaluation_chart, train_models,
)


def main():
    raw_rows = len(pd.read_csv(DATA_PATH))
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_eda_charts(data)
    models, metrics, X_test, y_test, predictions = train_models(data)
    save_evaluation_chart(y_test, predictions)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    best_model_name = metrics.iloc[0]["Model"]
    joblib.dump(models[best_model_name], OUTPUT_DIR / "best_model.joblib")
    linear_coefficients(models["Linear Regression"]).to_csv(
        OUTPUT_DIR / "linear_coefficients.csv", index=False
    )
    print(f"Rows used after cleaning: {len(data)}")
    print(f"Rows removed during cleaning: {raw_rows - len(data)}")
    print("Holdout metrics (predictions compared with actual observed sales):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Linear regression coefficients (sales units per advertising spend unit):")
    print(linear_coefficients(models["Linear Regression"]).to_string(index=False))
    print(f"Charts and result tables saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
