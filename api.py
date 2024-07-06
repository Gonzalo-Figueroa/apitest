import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# Cargar el modelo guardado
loaded_model = joblib.load('logistic_model_balanced.pkl')

# Inicializar la aplicación FastAPI
app = FastAPI()

# Definir el modelo de datos
class Data(BaseModel):
    features: list

# Definir el endpoint para hacer predicciones
@app.post("/predict")
def predict(data: Data):
    features = np.array(data.features).reshape(1, -1)
    prediction = loaded_model.predict(features)
    return {"prediction": prediction.tolist()}

@app.get("/")
def read_root():
    return {"message": "API de predicciones con FastAPI"}
