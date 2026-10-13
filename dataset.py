import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).parent

categories = {
    "Billing": [
        "I was charged twice",
        "My refund has not arrived",
        "Incorrect amount on my invoice",
        "I need help with my payment",
        "My subscription payment failed"
    ],
    "Technical": [
        "The application keeps crashing",
        "I cannot upload my file",
        "The website is very slow",
        "I get an error when logging in",
        "The application is not working"
    ],
    "Account": [
        "I cannot reset my password",
        "Please update my email address",
        "My account is locked",
        "I need to change my username",
        "I cannot access my account"
    ],
    "Shipping": [
        "My order has not arrived",
        "Where is my package",
        "My delivery is delayed",
        "I received the wrong item",
        "My package arrived damaged"
    ]
}

records = []

for category, messages in categories.items():
    for i in range(125):
        text = messages[i % len(messages)]
        priority = (
            "High" if i % 3 == 0
            else "Medium" if i % 3 == 1
            else "Low"
        )
        records.append({
            "text": f"{text} Reference number {i + 1}",
            "category": category,
            "priority": priority
        })

df = pd.DataFrame(records)

df["text"] = df["text"].str.lower().str.strip()
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

df.to_csv(BASE_DIR / "tickets.csv", index=False)

print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts())

print("\nPriority counts:")
print(df["priority"].value_counts())

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["category"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["category"]
)

train_df.to_csv(BASE_DIR / "train.csv", index=False)
val_df.to_csv(BASE_DIR / "validation.csv", index=False)
test_df.to_csv(BASE_DIR / "test.csv", index=False)

print("\nTrain rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))
print("\nDataset and splits saved successfully!")