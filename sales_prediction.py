"""Beginner-friendly sales prediction utilities for the Advertising dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Advertising.csv"
OUTPUT_DIR = ROOT / "outputs"
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"
RANDOM_STATE = 42


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, clean, and validate the provided dataset."""
    data = pd.read_csv(path)
    # Drop the original row number stored as an unnamed CSV index column.
    data = data.loc[:, ~data.columns.astype(str).str.match(r"^Unnamed")]
    required = FEATURES + [TARGET]
    missing_columns = sorted(set(required) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    data = data[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.drop_duplicates().dropna(subset=required).reset_index(drop=True)
    if data.empty:
        raise ValueError("No complete rows remain after cleaning the dataset.")
    if (data[FEATURES] < 0).any().any():
        raise ValueError("Advertising budgets must not be negative.")
    return data


def train_models(data: pd.DataFrame):
    """Fit models on a reproducible holdout split and return metrics."""
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE
        ),
    }
    rows, predictions = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        rows.append({
            "Model": name,
            "R-squared": r2_score(y_test, predicted),
            "MAE": mean_absolute_error(y_test, predicted),
            "RMSE": np.sqrt(mean_squared_error(y_test, predicted)),
        })
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return models, metrics, X_test, y_test, predictions


def linear_coefficients(model) -> pd.DataFrame:
    """Return linear coefficients in the original advertising spend units."""
    scaler = model.named_steps["standardscaler"]
    regression = model.named_steps["linearregression"]
    return pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": regression.coef_ / scaler.scale_,
    }).sort_values("Coefficient", ascending=False)


def create_eda_charts(data: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    """Save EDA charts based on observed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, feature in zip(axes, FEATURES):
        sns.regplot(data=data, x=feature, y=TARGET, ax=axis,
                    scatter_kws={"alpha": 0.65}, line_kws={"color": "crimson"})
        axis.set_title(f"{feature} advertising vs observed sales")
    fig.tight_layout()
    fig.savefig(output_dir / "advertising_vs_sales.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="vlag", center=0,
                fmt=".2f", ax=axis)
    axis.set_title("Correlation between advertising and sales")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data[FEATURES].sum().plot(kind="bar", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Total observed advertising spend by channel")
    axes[0].set_ylabel("Spend (dataset units)")
    data[TARGET].plot(kind="hist", bins=18, ax=axes[1], color="#F58518", edgecolor="white")
    axes[1].set_title("Observed sales distribution")
    axes[1].set_xlabel("Sales (dataset units)")
    fig.tight_layout()
    fig.savefig(output_dir / "spend_and_sales_distribution.png", dpi=160)
    plt.close(fig)


def save_evaluation_chart(y_test, predictions, output_dir: Path = OUTPUT_DIR) -> None:
    """Save observed-versus-predicted test-set comparisons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(predictions), figsize=(6 * len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]
    low, high = float(y_test.min()), float(y_test.max())
    for axis, (name, predicted) in zip(axes, predictions.items()):
        axis.scatter(y_test, predicted, alpha=0.75)
        axis.plot([low, high], [low, high], color="crimson", linestyle="--")
        axis.set(xlabel="Actual observed sales", ylabel="Predicted sales",
                 title=f"{name}: actual vs predicted")
    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def budget_scenario(model, budgets: dict[str, float]) -> float:
    """Predict a single hypothetical budget scenario."""
    row = pd.DataFrame([[budgets[name] for name in FEATURES]], columns=FEATURES)
    if (row < 0).any().any():
        raise ValueError("Advertising budgets must not be negative.")
    return float(model.predict(row)[0])
