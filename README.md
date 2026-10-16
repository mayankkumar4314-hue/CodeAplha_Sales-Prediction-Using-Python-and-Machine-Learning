# Sales Prediction Using Python

A beginner-friendly regression project using the supplied Advertising dataset. It explores how TV, Radio, and Newspaper advertising budgets relate to recorded sales, compares two models, and provides a Streamlit dashboard for hypothetical estimates.

## Project files

- `data/Advertising.csv` — supplied dataset.
- `sales_prediction.py` — shared cleaning, model training, evaluation, and chart helpers.
- `run_analysis.py` — creates EDA charts and evaluation files.
- `app.py` — interactive dashboard.
- `outputs/` — generated charts, result tables, and fitted best-model artifact; refreshed by the analysis script.

## Dataset and preprocessing

The CSV has 200 rows and four relevant columns: `TV`, `Radio`, `Newspaper`, and `Sales`. Its leading unnamed column is the source row index, so the loader drops it. The dataset contains numeric channels only; it has no target-segment or platform category fields, so categorical encoding is not applicable. The loader converts required fields to numbers, drops duplicate and incomplete rows, and checks that advertising budgets are nonnegative. It does not fill missing values with invented data.

The CSV does not document currency or measurement units. This project retains its recorded values; predicted sales are in the dataset's sales units.

## Setup

Python 3.10 or newer is recommended. In the project folder:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. In Google Colab, upload the project (including `data/Advertising.csv`), then run `!pip install -r requirements.txt` in a notebook cell.

## Run the analysis

```bash
python run_analysis.py
```

This saves advertising-versus-sales scatter plots, a correlation heatmap, spend and sales charts, actual-versus-predicted plots, `outputs/model_metrics.csv`, `outputs/linear_coefficients.csv`, and `outputs/best_model.joblib` (the model with the lower test RMSE). The script uses a reproducible 80/20 train/test split, fits Linear Regression and Random Forest Regression, then reports R-squared, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on held-out observations. The dashboard trains from the CSV on startup and does not depend on the saved artifact.

### Results calculated from the supplied CSV

The cleaned CSV had **200 rows**; no rows were removed. On the fixed 20% test split (40 observations), the measured results were:

| Model | R-squared | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest | 0.980 | 0.662 | 0.791 |
| Linear Regression | 0.899 | 1.461 | 1.782 |

For this one split, Random Forest had lower errors. The fitted linear model's coefficients were Radio **0.189**, TV **0.045**, and Newspaper **0.003** predicted sales units per spend unit, holding other channels fixed. These are associations learned by the model, not proof that increasing a channel's budget causes that sales change. Metrics can vary with another split or new data.

## Run the dashboard

```bash
streamlit run app.py
```

Choose a model, inspect observed data and holdout evaluation, then enter channel budgets to see a **predicted** sales estimate. It is not an actual sale or guaranteed outcome. Keep inputs within the observed ranges where possible because the models have little evidence for extrapolation.

## How it works

1. **Load and clean:** keep three numeric advertising features and the sales target; discard the CSV index, duplicate rows, and incomplete rows.
2. **Explore:** review observed channel-versus-sales plots, total channel spend, the sales distribution, and feature correlations.
3. **Split:** hold out 20% of rows for evaluation.
4. **Train:** fit a scaled Linear Regression baseline and a Random Forest regressor on the training rows.
5. **Evaluate:** compare R², MAE, and RMSE against actual sales in the test set. Lower MAE/RMSE mean smaller errors; R² describes variance explained on this test split.
6. **Interpret:** inspect original-unit linear coefficients or Random Forest feature importance. These describe model associations/usefulness, not causal effects.
7. **Predict:** enter budgets for a hypothetical estimate, clearly labeled as predicted sales.

## Interpreting results and business use

Run the analysis to calculate results for this dataset and split. Use the metrics file to see which model performed better on held-out rows and the coefficient file to compare fitted per-unit associations. A larger positive coefficient does not automatically mean a channel deserves more budget: costs, profit margins, uncertainty, and causal evidence also matter. Treat patterns as exploratory and test changes before reallocating a real budget.

This dataset has no dates, customer segments, campaign outcomes, or profit data. It cannot support time-series forecasts, segment comparisons, profit optimization, or causal claims. Results can vary with a different split or new data.

## Conclusion

This project demonstrates a complete small regression workflow, from an actual CSV through EDA, evaluation, interpretation, and an interactive prediction interface. Use computed holdout metrics and plots to discuss the observed dataset; do not present a model prediction as actual sales or guaranteed business impact.
