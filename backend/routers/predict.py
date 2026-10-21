from fastapi import APIRouter

from backend.schemas.prediction import (
    PredictionRequest,
    PredictionResponse
)

router = APIRouter()


@router.post(
    "/predictions",
    response_model=PredictionResponse
)
def predict(request: PredictionRequest):
    ...