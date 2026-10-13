from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Customer Support Ticket Classifier")

category_model = joblib.load(BASE_DIR / "category_model.joblib")
priority_model = joblib.load(BASE_DIR / "priority_model.joblib")


class TicketRequest(BaseModel):
    text: str = Field(min_length=1, description="Support ticket text")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: TicketRequest):
    text = request.text.strip()

    if not text:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Ticket text cannot be empty"
        )

    category = category_model.predict([text])[0]
    priority = priority_model.predict([text])[0]

    category_conf = max(category_model.predict_proba([text])[0])
    priority_conf = max(priority_model.predict_proba([text])[0])

    confidence = float((category_conf + priority_conf) / 2)

    return {
        "category": str(category),
        "priority": str(priority),
        "confidence": round(confidence, 2)
    }

