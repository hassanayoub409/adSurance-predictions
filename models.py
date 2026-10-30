"""Model training and prediction for the two projects.

Both models are scikit-learn Pipelines, so scaling is applied automatically
at prediction time (no manual scaler bookkeeping in the app).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA_DIR = Path(__file__).parent / "data"

# Category codes used in insurance.csv
SEX_CODES = {"Female": 0, "Male": 1}
REGION_CODES = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}

MEDICAL_FEATURES = ["age", "sex", "bmi", "children", "smoker", "region"]
AD_FEATURES = [
    "Age",
    "Daily_Internet_Usage",
    "Daily_Time_Spent_on_Site",
    "Ad_Topic_Score",
    "Area_Income",
    "Timestamp_Hour",
]


@dataclass
class TrainedModel:
    pipeline: Pipeline
    features: list
    metrics: dict  # evaluated on the held-out 20% test split


def train_medical_model() -> TrainedModel:
    """Polynomial (degree 2) linear regression predicting insurance charges."""
    data = pd.read_csv(DATA_DIR / "insurance.csv")
    X, y = data[MEDICAL_FEATURES], data["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    ).fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {
        "R²": r2_score(y_test, pred),
        "MAE ($)": mean_absolute_error(y_test, pred),
    }
    return TrainedModel(pipeline, MEDICAL_FEATURES, metrics)


def train_ad_model() -> TrainedModel:
    """Logistic regression predicting whether a user clicks an ad."""
    data = pd.read_csv(DATA_DIR / "ad_click_dataset.csv")
    X, y = data[AD_FEATURES], data["Clicked"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("model", LogisticRegression())]
    ).fit(X_train, y_train)
    metrics = {"Accuracy": accuracy_score(y_test, pipeline.predict(X_test))}
    return TrainedModel(pipeline, AD_FEATURES, metrics)


def predict_charges(model: TrainedModel, age, sex, bmi, children, smoker, region) -> float:
    row = pd.DataFrame(
        [[age, SEX_CODES[sex], bmi, children, int(smoker), REGION_CODES[region]]],
        columns=model.features,
    )
    # Linear regression can go slightly negative at the extremes; clamp at 0.
    return max(0.0, float(model.pipeline.predict(row)[0]))


def predict_click(model: TrainedModel, age, internet_usage, time_on_site,
                  topic_score, area_income, hour) -> float:
    """Return the probability (0-1) that the user clicks the ad."""
    row = pd.DataFrame(
        [[age, internet_usage, time_on_site, topic_score, area_income, hour]],
        columns=model.features,
    )
    return float(model.pipeline.predict_proba(row)[0, 1])
