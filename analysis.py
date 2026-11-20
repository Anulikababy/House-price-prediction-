"""Exploratory analysis and Linear Regression evaluation for Project 2."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
df = X.copy()
df["MedHouseVal"] = y

print("Dataset shape:", df.shape)
print("\\nMissing values:\\n", df.isna().sum())
correlations = df.corr(numeric_only=True)["MedHouseVal"].sort_values(ascending=False)
print("\\nCorrelation with target:\\n", correlations)
correlations.to_csv(OUT / "feature_correlations.csv", header=["correlation"])

plt.figure(figsize=(10, 7))
plt.imshow(df.corr(numeric_only=True), cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(df.columns)), df.columns, rotation=75, ha="right")
plt.yticks(range(len(df.columns)), df.columns)
plt.title("California Housing Feature Correlation")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close()

# The dataset's eight numeric features are retained; scaling is performed in a pipeline.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"\\nMAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred, alpha=0.25)
plt.xlabel("Actual median house value ($100,000 units)")
plt.ylabel("Predicted median house value ($100,000 units)")
plt.title("Actual vs Predicted House Values")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

pd.DataFrame({"actual": y_test, "predicted": pred}).to_csv(
    OUT / "predictions.csv", index=False
)
print("\\nFiles saved in the outputs/ folder.")
