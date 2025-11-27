from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
import os
from models.models import model_predict
import joblib
from os.path import join as join

class LoanData(BaseModel):
    age: str
    taille: str
    poids: str
    sexe: str
    sport_licence: str
    niveau_etude: str
    region: str
    smoker: str
    nationalite_francaise: str
    revenu_estime_mois: str

#logger.add("logs/api.log", rotation="500 MB", level="DEBUG")

app = FastAPI()
#model = joblib.load(join("models", "v2025_11_combined_ts10_rs42"))

@app.post("/predict/")
async def predict(payload):
    try:
#        logger.info(f"Received data: {payload}")
        print(payload)
        
        pred = model_predict(model, payload)

        return { pred }
    except Exception as e:
        print(e)
        logger.error(f"Erreur lors de l'analyse: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=True)
