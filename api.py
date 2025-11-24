from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
import torch
import os
from models.models import model_predict


class Texte(BaseModel):
    texte: str


model = joblib.load(join("models", "v2025_11_combined_ts10_rs42"))


logger.add("logs/api.log", rotation="500 MB", level="INFO")

app = FastAPI()


@app.post("/predict/")
async def predict(payload: Texte):
    logger.info(f"Received text: {payload.texte}")

    try:
        pred = model_predict(model, payload)

        return {pred}
    except Exception as e:
        print(e)
        logger.error(f"Erreur lors de l'analyse: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=9000, reload=True)
