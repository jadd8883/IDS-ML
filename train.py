
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/network_traffic.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ids_model.joblib"

# 1. Load and clean the dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

# Remove duplicate column names, if any
df = df.loc[:, ~df.columns.duplicated()]

if "Label" not in df.columns:
    raise ValueError(
        "Label column not found. Rename your target column to 'Label'."
    )

df["Label"] = df["Label"].astype(str).str.strip()

# Convert the dataset into a binary classification problem
labels = df["Label"].str.upper()
y = (labels != "BENIGN").astype(int)

print("Total records:", len(df))
print("Class distribution:")
print(y.value_counts().rename({0: "Benign", 1: "Attack"}))

# 2. Select numeric network-flow features
# Exclude labels, identifiers, IP addresses and timestamps.
excluded = {
    "label", "flow id", "source ip", "destination ip",
    "timestamp", "sourceip", "destinationip"
}

feature_df = df.drop(
    columns=[
        col for col in df.columns
        if col.strip().lower() in excluded
    ],
    errors="ignore"
)

X = feature_df.select_dtypes(include=[np.number]).copy()

if X.shape[1] == 0:
    raise ValueError("No numeric features found in the CSV.")

# Convert infinite values into missing values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove rows with no usable feature values
valid = X.notna().any(axis=1)
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Remove constant features
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] == 0:
    raise ValueError("No useful numeric features remain.")

if y.nunique() < 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Need at least two classes with two records per class."
    )

# 3. Split data before fitting the model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Build a preprocessing + classification pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

# 5. Train
print("\nTraining model...")
model.fit(X_train, y_train)

# 6. Evaluate on held-out records
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(
    y_test, predictions, zero_division=0
), 4))
print("Recall:", round(recall_score(
    y_test, predictions, zero_division=0
), 4))
print("F1 score:", round(f1_score(
    y_test, predictions, zero_division=0
), 4))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Attack"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 7. Save the model and its feature schema
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "features": X.columns.tolist(),
}, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Features used:", len(X.columns))
