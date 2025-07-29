import os
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))


class PodPredictor:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            self.model_dir = str(Path(__file__).parent / "saved_models")
        else:
            self.model_dir = model_dir
            
        self.models = self._load_models()
    
    def _load_models(self) -> Dict:
        try:
            models = {}
            model_files = [f for f in os.listdir(self.model_dir) 
                          if f.endswith('_model.joblib')]
            
            for model_file in model_files:
                target = model_file.replace('_model.joblib', '')
                model_path = os.path.join(self.model_dir, model_file)
                models[target] = joblib.load(model_path)
            
            return models
        except Exception as e:
            # Log error but don't crash
            print(f"Error loading models: {str(e)}")
            return {}
    
    def _prepare_features(self, 
                         date: Union[str, datetime],
                         gmv: float,
                         users: int,
                         marketing_cost: float) -> pd.DataFrame:
        """
        Prepare features for prediction
        
        Args:
            date: Date for prediction
            gmv: Gross Merchandise Value
            users: Number of users
            marketing_cost: Marketing cost
            
        Returns:
            DataFrame of features ready for prediction in the exact same order as during training
        """
        # Convert string date to datetime if needed
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        import numpy as np
            
        # Step 1: Create base features first (as in the original training data)
        features = pd.DataFrame({
            'gmv': [gmv],
            'users': [users],
            'marketing_cost': [marketing_cost],
            'day_of_week': [date.weekday()],
            'week_of_year': [date.isocalendar()[1]],
            'month': [date.month],
            'day_of_month': [date.day],
            'is_weekend': [1 if date.weekday() >= 5 else 0]
        })
        
        # Step 2: Create enhanced features in the exact same order as in train.py
        # This is critical for the model to work correctly
        
        # Per user metrics
        features['gmv_per_user'] = gmv / users if users > 0 else 0
        features['marketing_per_user'] = marketing_cost / users if users > 0 else 0
        
        # User transformations
        features['user_sqrt'] = np.sqrt(users)
        features['user_log'] = np.log1p(users)
        features['user_squared'] = np.power(users, 2)
        
        # Combined metrics
        features['user_gmv_product'] = users * gmv
        features['user_to_gmv_ratio'] = users / gmv if gmv > 0 else 0
        
        # Traffic features
        features['traffic_index'] = users * (gmv / users if users > 0 else 0)
        features['peak_factor'] = users * (1 + marketing_cost / (gmv + 1))
        
        # Time-based interaction features
        features['weekend_users'] = users * features['is_weekend'].iloc[0]
        features['day_user_interaction'] = users * features['day_of_week'].iloc[0]
        
        # Normalized features
        features['users_normalized'] = 0.0  # Since we can't calculate mean/std for a single prediction
        features['gmv_normalized'] = 0.0
        features['marketing_cost_normalized'] = 0.0
        
        return features
    
    def predict(self, 
               date: Union[str, datetime],
               gmv: float,
               users: int,
               marketing_cost: float) -> Dict[str, int]:
        """
        Make pod consumption predictions
        
        Args:
            date: Date for prediction
            gmv: Gross Merchandise Value
            users: Number of users
            marketing_cost: Marketing cost
            
        Returns:
            Dictionary of predictions for each target
        """
        try:
            # Check if models are loaded
            if not self.models:
                raise ValueError("No models loaded. Please ensure models are trained and saved.")
                
            # Prepare features
            features = self._prepare_features(date, gmv, users, marketing_cost)
            
            # Make predictions for each target
            predictions = {}
            for target, model in self.models.items():
                pred = model.predict(features)[0]
                # Round to nearest integer and ensure minimum of 1
                predictions[target] = max(1, round(pred))
            
            return predictions
        except Exception as e:
            # Log error and return a default prediction
            print(f"Error making prediction: {str(e)}")
            return {"fe_pods": None, "be_pods": None}


if __name__ == "__main__":
    # Test prediction
    predictor = PodPredictor()
    
    # Example prediction
    prediction = predictor.predict(
        date="2024-05-01",
        gmv=10000000.0,
        users=80000,
        marketing_cost=150000.0
    )
    
    print("Prediction:")
    for target, value in prediction.items():
        print(f"  {target}: {value}")
