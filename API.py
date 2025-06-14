# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("logreg_model_optimise.joblib")  # Le fichier doit exister ici

FEATURES = ['ca', 'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'oldpeak']

class PatientData(BaseModel):
    ca: int
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    oldpeak: float
    

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API de prédiction de maladie cardiaque prête !"}

@app.post("/predict")
def predict(data: PatientData):
    input_df = pd.DataFrame([data.dict()])[FEATURES]
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df).max()
    return {
        "prediction": int(prediction),
        "confidence": round(float(proba), 4)
    }
