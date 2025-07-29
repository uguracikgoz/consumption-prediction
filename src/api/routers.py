from fastapi import APIRouter, HTTPException, Depends
from typing import List
from datetime import datetime

from api.models import PredictionRequest, PredictionResponse
from api.services import PredictionService


# Global dependency function
def get_prediction_service() -> PredictionService:
    """Dependency to get the prediction service instance"""
    return PredictionService()


class PredictionRouter:
    """Router class for handling prediction endpoints"""
    
    def __init__(self):
        self.router = APIRouter(tags=["predictions"])
        self._setup_routes()
    
    def _setup_routes(self):
        self.router.post(
            "/predict",
            response_model=PredictionResponse
        )(self.predict)
        
        self.router.post(
            "/batch-predict"
        )(self.batch_predict)
    
    async def predict(self, request: PredictionRequest, service: PredictionService = Depends(get_prediction_service)):
        """
        Predict pod consumption based on business metrics
        
        Args:
            request: PredictionRequest with business metrics
            service: PredictionService instance from dependency
            
        Returns:
            PredictionResponse with pod predictions
        """
        try:
            # Parse date
            try:
                # Validation happens here
                datetime.strptime(request.date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, 
                                   detail="Invalid date format. Please use YYYY-MM-DD.")
            
            # Make prediction
            prediction = service.predict(
                date_str=request.date,
                gmv=request.gmv,
                users=request.users,
                marketing_cost=request.marketing_cost
            )
            
            # Ensure we have values for both pod types
            if prediction.get("fe_pods") is None or prediction.get("be_pods") is None:
                raise HTTPException(status_code=500, 
                                   detail="Prediction failed. Models may not be properly loaded.")
            
            return PredictionResponse(**prediction)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, 
                               detail=f"Prediction error: {str(e)}")
    
    async def batch_predict(self,
                          requests: List[PredictionRequest],
                          service: PredictionService = Depends(get_prediction_service)):
        """
        Batch predict pod consumption for multiple dates/inputs
        
        Args:
            requests: List of PredictionRequest objects
            service: PredictionService instance from dependency
            
        Returns:
            List of predictions
        """
        try:
            return service.batch_predict(requests)
        except Exception as e:
            raise HTTPException(status_code=500, 
                               detail=f"Batch prediction error: {str(e)}")


# Create router instance
prediction_router = PredictionRouter().router


class SystemRouter:
    """Router class for system endpoints"""
    
    def __init__(self):
        self.router = APIRouter(tags=["system"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup router endpoints"""
        self.router.get("/")(self.root)
        self.router.get("/health")(self.health_check)
    
    async def root(self):
        """Root endpoint"""
        return {
            "message": "Pod Consumption Prediction API",
            "docs": "/docs",
            "status": "active"
        }
    
    async def health_check(self, service: PredictionService = Depends(get_prediction_service)):
        """Health check endpoint"""
        return {
            "status": "healthy",
            "redis": "connected" if service.cache.is_connected() else "disconnected"
        }

system_router = SystemRouter().router