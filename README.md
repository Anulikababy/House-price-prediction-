# Project 2: House Price Prediction (Regression)

## Goal
Predict California median house values from eight housing features using Linear Regression.

## Dataset
The California Housing dataset contains 20,640 samples and 8 input features. It is loaded through `sklearn.datasets.fetch_california_housing`. On the first run, scikit-learn downloads the dataset, so an internet connection is required.

Dataset documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

## Project activities
- Inspect the data and check missing values.
- Examine correlations between features and the target.
- Use all eight numeric features as predictors.
- Split data into training and testing sets.
- Impute missing values if present, scale features, and train Linear Regression.
- Evaluate with Mean Absolute Error (MAE) and R².
- Plot a correlation heatmap and actual-versus-predicted values.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
streamlit run app.py
```

## Streamlit app
The app accepts the eight feature values and returns an estimated median house value. The displayed price is converted from the dataset's $100,000 units into dollars. The app trains and caches the model on its first launch.

## Deploy on Streamlit Community Cloud
1. Create a public GitHub repository and upload all files in this folder.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy. Copy the generated app URL for submission.

## Important
The dataset is historical and the model is for learning purposes. Predictions are estimates, not official property valuations.
