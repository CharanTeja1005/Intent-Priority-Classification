from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-small")

class ClassificationRequest(BaseModel):
    text: str
    labels: list[str]

class PriorityRequest(BaseModel):
    text: str
    priority_labels: list[str]

@app.post("/classify")
def classify_text(request: ClassificationRequest):
    try:
        result = classifier(request.text, request.labels)
        best_label = result['labels'][0]
        return {"best_label": best_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prioritize")
def prioritize_text(request: PriorityRequest):
    try:
        priority_result = classifier(request.text, request.priority_labels)
        priority = priority_result['labels'][0]
        return {"priority": priority}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
