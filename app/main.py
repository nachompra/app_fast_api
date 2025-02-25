from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = FastAPI()

with open("data/advertising_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


class PredictRequest(BaseModel):
    tv: float
    radio: float
    newspaper: float

@app.get("/")
def home():
    return {"message": "API Funcionando"}


# 1. Endpoint de predicción
@app.post("/v1/predict")
async def predict(request: PredictRequest):
    tv = request.tv
    radio = request.radio
    newspaper = request.newspaper

    prediction = model.predict([[tv, radio, newspaper]])[0]
    return {"Sales prediction": round(prediction, 2)}


# 2. Endpoint de ingesta de datos
class Invest(BaseModel):
    TV: float
    newspaper: float
    radio: float
    sales: float


@app.post("/ingest")
async def add_invest(advertising: Invest):
    with sqlite3.connect('data/advertising.db') as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO advertising (TV, newspaper, radio, sales) VALUES (?, ?, ?, ?)",
                (advertising.TV, advertising.newspaper, advertising.radio, advertising.sales)
            )
            conn.commit() 
            invest_id = cursor.lastrowid 
        except Exception as e:
            conn.rollback() 
            raise HTTPException(status_code=500, detail=f"Error inserting investment: {str(e)}")
        
        return {'message': 'Datos ingresados correctamente'}


# 3. Endpoint de reentramiento del modelo

with open("data/advertising_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# MAE base del modelo en producción
MAE_BASE = 100
THRESHOLD = 1.2  # 20% superior a MAE_BASE

@app.post("/retrain")
async def retrain_model():
    try:
        # Cargamos BBDD
        with sqlite3.connect('data/advertising.db') as conn:
            query = "SELECT TV, radio, newspaper, sales FROM advertising"
            df = pd.read_sql(query, conn)


        # Dividimos el train/test
        df_train = df.iloc[:80]  # Primeros 80 registros
        df_test = df.iloc[80:100]  # Siguientes 20 registros (test del modelo actual)
        df_new = df.iloc[100:]  # 30 nuevos registros

        # Evaluamos modelo con los nuevos datos
        X_new = df_new[["TV", "radio", "newspaper"]]
        y_new = df_new["sales"]
        y_pred_new = model.predict(X_new)
        new_mae = mean_absolute_error(y_new, y_pred_new)

        # Check para que existan suficientes datos
        if df.shape[0] < 100:  
            raise HTTPException(status_code=400, detail="No hay suficientes datos para evaluar el modelo.")

        # Dividimos para entreno
        X = df[["TV", "radio", "newspaper"]]
        y = df["sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # Check si sigue generalizando bien
        if new_mae <= MAE_BASE * THRESHOLD:
            return {"message": "El modelo sigue siendo válido. No es necesario reentrenar.", "new_MAE": new_mae}

        # Reentrenamos en caso de que el MAE empeore con nuevos datos
        X_train = df_train[["TV", "radio", "newspaper"]]
        y_train = df_train["sales"]
        new_model = LinearRegression()
        new_model.fit(X_train, y_train)

        # Evaluamos con el nuevo entreno
        y_pred_new_retrain = new_model.predict(X_new)
        retrained_mae = mean_absolute_error(y_new, y_pred_new_retrain)

        # Check de MAE
        if retrained_mae > MAE_BASE * THRESHOLD:
            return {"message": "El modelo reentrenado sigue sin generalizar bien. Considera una nueva modelización.", "retrained_MAE": retrained_mae}

        # Modelo Actualizado
        with open("data/advertising_model.pkl", "wb") as model_file:
            pickle.dump(new_model, model_file)

        return {"message": "Modelo reentrenado con éxito.", "retrained_MAE": retrained_mae}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el reentrenamiento: {str(e)}")
