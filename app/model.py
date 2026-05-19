
import joblib
from app.config import models_path

model=joblib.load(models_path)

def predict_text(message: str) -> dict:

    label = model.predict([message])[0]
    probs = model.predict_proba([message])[0]
    classes = model.classes_
    idx = list(classes).index(label)
    score = float(probs[idx])


    return {

        "label":label,
        "score" : score,
    }

        

