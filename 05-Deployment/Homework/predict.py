import pickle
from fastapi import FastAPI
from typing import Dict, Any
import uvicorn

from typing import Literal
from pydantic import BaseModel, Field

from pydantic import ConfigDict


class Lead(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_source: str
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    convert_probability: float
    will_convert: bool

app = FastAPI(title="lead-conversion-prediction")

with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(lead):
    result = pipeline.predict_proba([lead])[0, 1]
    return float(result)


@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    prob = predict_single(lead.model_dump())

    return PredictResponse(
        convert_probability=round(prob, 3),
        will_convert=prob >= 0.5
    )


@app.get("/")
def root():
    return {"message": "Lead Conversion Prediction API"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9697)