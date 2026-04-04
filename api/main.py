from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_from_geojson


app = FastAPI(title="Urban Resilience Engine")


class FarmInput(BaseModel):
    geojson: dict
    year: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "message": "happy coding"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": "1.0.0"}


@app.post("/predict")
def predict(payload: FarmInput) -> dict:
    return predict_from_geojson(payload.geojson, payload.year)
