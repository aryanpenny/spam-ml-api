from pathlib import Path
import os

models_path=  Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

