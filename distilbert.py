import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

BASE_DIR = Path(__file__).parent
MODEL_NAME = "distilbert-base-uncased"

train_df = pd.read_csv(BASE_DIR / "train.csv")
val_df = pd.read_csv(BASE_DIR / "validation.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")

labels = sorted(train_df["category"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

for df in [train_df, val_df, test_df]:
    df["label"] = df["category"].map(label2id)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def prepare_dataset(df):
    dataset = Dataset.from_pandas(
        df[["text", "label"]].reset_index(drop=True)
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset

train_ds = prepare_dataset(train_df)
val_ds = prepare_dataset(val_df)
test_ds = prepare_dataset(test_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_f1": f1_score(
            true_labels, predictions, average="macro"
        )
    }

args = TrainingArguments(
    output_dir=str(BASE_DIR / "distilbert_results"),
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    report_to="none",
    save_total_limit=1,
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

start = time.time()
trainer.train()
training_time = time.time() - start

print("Validation results:",trainer.evaluate(val_ds))
print("Test results:",trainer.evaluate(test_ds))

trainer.save_model(str(BASE_DIR / "distilbert_model"))
tokenizer.save_pretrained(str(BASE_DIR / "distilbert_model"))

with open(BASE_DIR / "category_labels.json", "w") as f:
    json.dump(label2id, f, indent=2)

print("\nTraining time:", round(training_time, 2), "seconds")
