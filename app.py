"""Interactive Streamlit dashboard for the Advertising sales example."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sales_prediction import (
    FEATURES, TARGET, budget_scenario, linear_coefficients, load_data, train_models,
)

st.set_page_config(page_title="Sales Prediction Using Python", page_icon="📈", layout="wide")
st.title("📈 Sales Prediction Using Python")
st.caption("Explore observed advertising data, compare models, and try hypothetical budget scenarios.")

try:
    data = load_data()
except (OSError, ValueError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

models, metrics, X_test, y_test, predictions = train_models(data)
selected_name = st.sidebar.selectbox("Prediction model", list(models))
selected_model = models[selected_name]

st.subheader("Dataset at a glance")
st.write(f"The cleaned dataset contains **{len(data)} rows**. Budgets and sales use the units recorded in the CSV.")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Model evaluation on held-out observations")
st.caption("These metrics compare predictions with actual observed sales in the 20% test split; they are not training-set scores.")
st.dataframe(metrics.style.format({"R-squared": "{:.3f}", "MAE": "{:.3f}", "RMSE": "{:.3f}"}), use_container_width=True)

left, right = st.columns(2)
with left:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(y_test, predictions[selected_name], alpha=0.75)
    bounds = [float(y_test.min()), float(y_test.max())]
    axis.plot(bounds, bounds, "r--", label="Perfect prediction")
    axis.set(xlabel="Actual observed sales", ylabel="Predicted sales", title=f"{selected_name}: actual vs predicted")
    axis.legend()
    st.pyplot(fig)
    plt.close(fig)
with right:
    fig, axis = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="vlag", center=0, fmt=".2f", ax=axis)
    axis.set_title("Correlation in observed dataset")
    st.pyplot(fig)
    plt.close(fig)

st.subheader("Observed advertising relationships")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for axis, feature in zip(axes, FEATURES):
    sns.regplot(data=data, x=feature, y=TARGET, ax=axis, scatter_kws={"alpha": 0.6}, line_kws={"color": "crimson"})
    axis.set_title(f"{feature} spend vs observed sales")
st.pyplot(fig)
plt.close(fig)

st.subheader("Feature contribution")
if selected_name == "Linear Regression":
    coefficients = linear_coefficients(selected_model)
    st.caption("A coefficient estimates the change in predicted sales for one additional spend unit, holding other channels fixed. It describes this fitted model and is not proof of causation.")
    st.dataframe(coefficients.style.format({"Coefficient": "{:.4f}"}), use_container_width=True)
else:
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": selected_model.feature_importances_}).sort_values("Importance", ascending=False)
    st.caption("Random Forest importance describes each feature's usefulness to this fitted model; it does not show causal impact.")
    st.bar_chart(importance.set_index("Feature"))

st.subheader("Try a hypothetical advertising budget")
st.caption("Use the same units as the dataset. Estimates are most grounded within the observed budget ranges. The result is a model prediction, not an actual sale or guarantee.")
inputs = {}
columns = st.columns(len(FEATURES))
for column, feature in zip(columns, FEATURES):
    inputs[feature] = column.number_input(
        f"{feature} budget", min_value=0.0,
        max_value=float(data[feature].max()),
        value=float(data[feature].median()), step=1.0,
    )
if st.button("Predict sales", type="primary"):
    estimate = budget_scenario(selected_model, inputs)
    st.success(f"Predicted sales: **{estimate:.2f}** dataset sales units")
    st.write("This is a hypothetical model estimate. The dataset contains no actual outcome for this scenario.")

st.subheader("Advertising impact analysis")
st.write("Linear coefficients provide a simple within-model comparison of channel associations. Read them alongside the observed scatter plots and test metrics; this dataset does not establish that changing a budget will cause the predicted change.")
impact_model = models["Linear Regression"]
fig, axis = plt.subplots(figsize=(8, 4))
baseline = {feature: float(data[feature].median()) for feature in FEATURES}
for feature in FEATURES:
    spend_values = pd.Series([data[feature].min(), data[feature].max()])
    scenarios = pd.DataFrame([baseline.copy(), baseline.copy()])
    scenarios[feature] = spend_values.to_numpy()
    estimates = impact_model.predict(scenarios[FEATURES])
    axis.plot(spend_values, estimates, marker="o", label=feature)
axis.set(xlabel="Advertising spend (dataset units)", ylabel="Predicted sales (dataset units)",
         title="Predicted sales as one channel varies (others held at their median)")
axis.legend(title="Channel")
st.pyplot(fig)
plt.close(fig)
st.caption("Scenario lines show model predictions over the observed minimum-to-maximum budget range, holding the other channels at their medians. This is a model-based comparison, not evidence of a causal or guaranteed sales change.")
