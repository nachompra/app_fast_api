from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import List
from sklearn.pipeline import Pipeline

app = FastAPI()

with open("data/advertising_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


class PredictRequest(BaseModel):
    data: List[List[float]]  

@app.get("/")
def home():
    return {"message": "API Funcionando"}


# 1. Endpoint de predicción
@app.post("/v1/predict")
async def predict(request: PredictRequest):
    try:
        for entry in request.data:
            if len(entry) != 3:
                raise HTTPException(status_code=400, detail="Cada entrada debe tener exactamente 3 valores (tv, radio, newspaper).")
        
        predictions = model.predict(request.data)
        
        return {"Sales prediction": [round(pred, 2) for pred in predictions]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")

# 2. Endpoint de ingesta de datos

class Invest(BaseModel):
    TV: float
    newspaper: float
    radio: float
    sales: float

@app.post("/ingest")
async def add_invest(data: dict):
    investments = [
        Invest(TV=entry[0], newspaper=entry[1], radio=entry[2], sales=entry[3]) 
        for entry in data['data']
    ]

    with sqlite3.connect('data/advertising.db') as conn:
        cursor = conn.cursor()
        try:
            cursor.executemany(
                "INSERT INTO advertising (TV, newspaper, radio, sales) VALUES (?, ?, ?, ?)",
                [(inv.TV, inv.newspaper, inv.radio, inv.sales) for inv in investments]
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Error inserting investments: {str(e)}")

    return {'message': 'Datos ingresados correctamente'}

# 3. Endpoint de reentramiento del modelo
# MAE base del modelo en producción
MAE_BASE = 100
THRESHOLD = 1.2

# Cargar el modelo previamente guardado
with open("data/advertising_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.post("/retrain")
async def retrain_model():
    try:
        # Cargamos BBDD
        with sqlite3.connect('data/advertising.db') as conn:
            query = "SELECT TV, radio, newspaper, sales FROM advertising"
            df = pd.read_sql(query, conn)

        # Check para que existan suficientes datos
        if df.shape[0] < 100:  
            raise HTTPException(status_code=400, detail="No hay suficientes datos para evaluar el modelo.")

        # Dividimos en train/test
        X = df[["TV", "radio", "newspaper"]]
        y = df["sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # Si el modelo cargado es un Pipeline, el pipeline se encargará de las transformaciones internamente
        if isinstance(model, Pipeline):
            # Aplicamos el pipeline directamente para hacer la predicción
            y_pred = model.predict(X_test)
        else:
            # Si no es un pipeline, usamos el modelo de forma tradicional
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Evaluamos el modelo con los nuevos datos
        new_mae = mean_absolute_error(y_test, y_pred)

        # Check si sigue generalizando bien
        if new_mae <= MAE_BASE * THRESHOLD:
            return {"message": "El modelo sigue siendo válido. No es necesario reentrenar."}

        # Reentrenamos el modelo
        new_model = LinearRegression()
        new_model.fit(X_train, y_train)

        # Evaluamos con el nuevo entrenamiento
        y_pred_retrain = new_model.predict(X_test)
        retrained_mae = mean_absolute_error(y_test, y_pred_retrain)

        # Check de MAE
        if retrained_mae > MAE_BASE * THRESHOLD:
            return {"message": "El modelo reentrenado sigue sin generalizar bien. Considera una nueva modelización."}

        # Guardamos el modelo actualizado
        with open("data/advertising_model.pkl", "wb") as model_file:
            pickle.dump(new_model, model_file)

        return {"message": "Modelo reentrenado correctamente."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")