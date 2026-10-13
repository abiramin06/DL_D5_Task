import pandas as pd
import joblib
import time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).parent

train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df = pd.read_csv(BASE_DIR / "validation.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")

def train_model(label):
    X_train = train_df["text"].astype(str)
    y_train = train_df[label]

    X_val = val_df["text"].astype(str)
    y_val = val_df[label]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    params = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "model__C": [0.1, 1, 10]
    }

    search = GridSearchCV(
        pipeline,
        params,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    start = time.time()
    search.fit(X_train, y_train)
    training_time = time.time() - start

    model = search.best_estimator_
    predictions = model.predict(X_val)

    print(f"\n===== {label.upper()} RESULTS =====")
    print("Best parameters:", search.best_params_)
    print("Accuracy:", round(accuracy_score(y_val, predictions), 4))
    print("Macro F1:", round(f1_score(y_val, predictions, average="macro"), 4))
    print("\nClassification report:")
    print(classification_report(y_val, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions))
    print("Training time:", round(training_time, 2), "seconds")

    joblib.dump(model, BASE_DIR / f"{label}_model.joblib")
    print(f"Saved: {label}_model.joblib")

    return model

category_model = train_model("category")
priority_model = train_model("priority")

print("\nBaseline training completed!")
