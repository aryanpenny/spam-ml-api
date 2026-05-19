from pydantic import BaseModel

class PredictionRequest(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    label: str
    score: float