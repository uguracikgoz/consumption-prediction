from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date for prediction in YYYY-MM-DD format")
    gmv: float = Field(..., description="Gross Merchandise Value")
    users: int = Field(..., description="Number of users")
    marketing_cost: float = Field(..., description="Marketing cost")


class PredictionResponse(BaseModel):
    fe_pods: int = Field(..., description="Frontend pod prediction")
    be_pods: int = Field(..., description="Backend pod prediction")
    date: str = Field(..., description="Date of the prediction")
    cached: bool = Field(False, description="Whether the prediction was served from cache")


class BatchPredictionResult(BaseModel):
    fe_pods: Optional[int] = Field(None, description="Frontend pod prediction")
    be_pods: Optional[int] = Field(None, description="Backend pod prediction")
    date: str = Field(..., description="Date of the prediction")
    error: Optional[str] = Field(None, description="Error message, if any")
