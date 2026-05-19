from fastapi import FastAPI
from app.schemas import PredictionRequest,PredictionResponse
from app.model import predict_text

app = FastAPI(title="spam detection app")

@app.get("/health")
def health():
    return {"status" : "ok"}

@app.post("/predict", response_model = PredictionResponse)
def predict(req: PredictionRequest):
    result = predict_text(req.message)
    return result


