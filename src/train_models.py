import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import joblib

RANDOM_STATE = 42
DATA_PATH = "data/CICIDS2017.csv"
TARGET_COLUMN = "Label"

MODELS_DIR = "models"
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "improved_rf.joblib")


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("CICIDS2017.csv missing in data/")
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Column 'Label' missing from dataset.")
    return df


def prepare_features(df):
    df = df.copy()

    def map_label(x):
        s = str(x).strip().upper()
        if s in ("BENIGN", "NORMAL"):
            return 0
        return 1

    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(map_label)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # One-hot encode string columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def save_feature_columns(cols):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        for c in cols:
            f.write(c + "\n")


def load_feature_columns():
    with open(FEATURE_COLUMNS_PATH) as f:
        return [x.strip() for x in f.readlines()]


def train_iteration4_model():
    print("\nBase paper used NO machine learning model.")
    print("They relied entirely on manual inspection.")
    print("This Random Forest classifier is the FIRST ML enhancement (Iteration-4).\n")

    print("[*] Loading dataset...")
    df = load_dataset()

    print("[*] Preparing features...")
    X, y = prepare_features(df)
    cols = list(X.columns)
    save_feature_columns(cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("[*] Training Iteration-4 Improved Random Forest Model...")

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {c: w for c, w in zip(classes, weights)}

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weights
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }

    print("\n=== Iteration-4 Results ===")
    print(results)
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")
    print(f"Feature columns saved at: {FEATURE_COLUMNS_PATH}")

    return results


def load_iteration4_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the Iteration-4 model first.")
    model = joblib.load(MODEL_PATH)
    cols = load_feature_columns()
    return model, cols


if __name__ == "__main__":
    train_iteration4_model()