from datetime import datetime
from typing import Dict, Any

from api.cache import RedisCache
from model.predict import PodPredictor


class PredictionService:
    """Service for handling pod consumption predictions"""
    
    def __init__(self):
        self.predictor = PodPredictor()
        self.cache = RedisCache()
    
    def _generate_cache_key(self, date: str, gmv: float, users: int, marketing_cost: float) -> str:
        """Generate cache key for prediction request"""
        return f"prediction:{date}:{gmv}:{users}:{marketing_cost}"
    
    def predict(self, date_str: str, gmv: float, users: int, marketing_cost: float) -> Dict[str, Any]:
        """
        Make a pod consumption prediction with optional caching
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            gmv: Gross Merchandise Value
            users: Number of users
            marketing_cost: Marketing cost
            
        Returns:
            Dictionary with prediction results and cache status
        """
        # Parse date
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        
        # Try to get from cache first
        cache_key = self._generate_cache_key(formatted_date, gmv, users, marketing_cost)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            cached_result["cached"] = True
            return cached_result
        
        # Make prediction
        prediction = self.predictor.predict(
            date=parsed_date,
            gmv=gmv,
            users=users,
            marketing_cost=marketing_cost
        )
        
        # Create response
        response = {
            "fe_pods": prediction.get("fe_pods"),
            "be_pods": prediction.get("be_pods"),
            "date": formatted_date,
            "cached": False
        }
        
        # Cache result
        self.cache.set(cache_key, response)
        
        return response
    
    def batch_predict(self, requests: list) -> list:
        """
        Perform batch predictions
        
        Args:
            requests: List of prediction request objects
            
        Returns:
            List of prediction results
        """
        results = []
        for request in requests:
            try:
                prediction = self.predict(
                    date_str=request.date,
                    gmv=request.gmv,
                    users=request.users,
                    marketing_cost=request.marketing_cost
                )
                
                results.append({
                    "fe_pods": prediction.get("fe_pods"),
                    "be_pods": prediction.get("be_pods"),
                    "date": prediction.get("date"),
                    "error": None
                })
            except Exception as e:
                results.append({
                    "fe_pods": None,
                    "be_pods": None,
                    "date": request.date,
                    "error": str(e)
                })
        
        return results
