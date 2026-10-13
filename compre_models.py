import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline

BASE_DIR = Path(__file__).parent
test_df = pd.read_csv(BASE_DIR / "test.csv")

X_test = test_df["text"].astype(str)
y_test = test_df["category"]

results = []

baseline = joblib.load(BASE_DIR / "category_model.joblib")

start = time.perf_counter()
baseline_predictions = baseline.predict(X_test)
baseline_time = (time.perf_counter() - start) / len(X_test)

results.append({
    "Model": "TF-IDF + Logistic Regression",
    "Accuracy": accuracy_score(y_test, baseline_predictions),
    "Macro F1": f1_score(y_test, baseline_predictions, average="macro"),
    "Training time": "See baseline.py output",
    "Inference time/ticket (seconds)": baseline_time
})

bert = pipeline(
    "text-classification",
    model=str(BASE_DIR / "distilbert_model"),
    tokenizer=str(BASE_DIR / "distilbert_model"),
    device=-1
)

start = time.perf_counter()
bert_predictions = [
    bert(text, truncation=True)[0]["label"]
    for text in X_test
]
bert_time = (time.perf_counter() - start) / len(X_test)

results.append({
    "Model": "DistilBERT",
    "Accuracy": accuracy_score(y_test, bert_predictions),
    "Macro F1": f1_score(y_test, bert_predictions, average="macro"),
    "Training time": "See distilbert.py output",
    "Inference time/ticket (seconds)": bert_time
})

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))
comparison.to_csv(BASE_DIR / "model_comparison.csv", index=False)

y_priority = test_df["priority"]

priority_baseline = joblib.load(
    BASE_DIR / "priority_model.joblib"
)

start = time.perf_counter()
priority_baseline_predictions = priority_baseline.predict(X_test)
priority_baseline_time = (
    time.perf_counter() - start
) / len(X_test)

results.append({
    "Model": "TF-IDF + Logistic Regression (Priority)",
    "Accuracy": accuracy_score(
        y_priority, priority_baseline_predictions
    ),
    "Macro F1": f1_score(
        y_priority,
        priority_baseline_predictions,
        average="macro"
    ),
    "Training time": "See baseline.py output",
    "Inference time/ticket (seconds)": priority_baseline_time
})
