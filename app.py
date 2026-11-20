import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 California House Price Prediction")
st.write("Enter housing-area characteristics to estimate the median house value.")

@st.cache_resource
def train_model():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, list(X.columns), mean_absolute_error(y_test, predictions), r2_score(y_test, predictions)

with st.spinner("Loading the California Housing dataset and training the model (first run only)..."):
    try:
        model, feature_names, mae, r2 = train_model()
    except Exception as exc:
        st.error("The California Housing dataset could not be downloaded. Check your internet connection and try again.")
        st.exception(exc)
        st.stop()

st.caption("Model: Linear Regression | Dataset: 20,640 California housing records")
c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
c2.metric("R² score", f"{r2:.3f}")
st.info("The target is expressed in units of $100,000. For example, a prediction of 2.5 represents approximately $250,000.")

defaults = {
    "MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 5.0, "AveBedrms": 1.0,
    "Population": 1000.0, "AveOccup": 3.0, "Latitude": 36.7, "Longitude": -119.5
}
labels = {
    "MedInc": "Median income (in tens of thousands of dollars)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Area population",
    "AveOccup": "Average occupants per household",
    "Latitude": "Latitude",
    "Longitude": "Longitude"
}
with st.form("prediction_form"):
    values = {}
    left, right = st.columns(2)
    for i, feature in enumerate(feature_names):
        col = left if i % 2 == 0 else right
        values[feature] = col.number_input(
            labels.get(feature, feature), value=float(defaults[feature]),
            format="%.3f"
        )
    submitted = st.form_submit_button("Predict house price", type="primary")

if submitted:
    row = pd.DataFrame([values], columns=feature_names)
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated median house value: ${prediction * 100000:,.0f}")
    st.caption("This is a model estimate, not a property valuation or guaranteed selling price.")
