#!/usr/bin/env python3
"""
Cron job to update predictions and cache them in Redis.
This script should be run once a day to generate predictions for upcoming days.
"""
import os
import sys
import json
import redis
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from model.predict import PodPredictor
from data.loader import DataLoader


class PredictionUpdater:
    def __init__(self, days_ahead=7):
        """
        Initialize the prediction updater
        
        Args:
            days_ahead: Number of days ahead to generate predictions for
        """
        self.days_ahead = days_ahead
        self.predictor = PodPredictor()
        
        # Initialize Redis
        self.redis_host = os.environ.get("REDIS_HOST", "localhost")
        self.redis_port = int(os.environ.get("REDIS_PORT", 6379))
        self.redis_password = os.environ.get("REDIS_PASSWORD", None)
        
        # Cache TTL in seconds (7 days)
        self.cache_ttl = 7 * 86400
        
        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                password=self.redis_password,
                decode_responses=True
            )
            # Test connection
            self.redis.ping()
            print("Redis connection established")
        except Exception as e:
            print(f"Redis connection failed: {str(e)}")
            self.redis = None
    
    def _load_latest_data(self):
        """
        Load and return the latest business metrics
        
        Returns:
            Dictionary with the latest business metrics
        """
        try:
            loader = DataLoader()
            df = loader.load_data()
            
            # Sort by date to get the latest data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Get the latest non-null values
            latest_data = {
                'gmv': df['gmv'].iloc[-1],
                'users': df['users'].iloc[-1],
                'marketing_cost': df['marketing_cost'].iloc[-1]
            }
            
            return latest_data
        except Exception as e:
            print(f"Error loading latest data: {str(e)}")
            
            # Return some default values if data loading fails
            return {
                'gmv': 10000000.0,
                'users': 80000,
                'marketing_cost': 150000.0
            }
    
    def update_predictions(self):
        """
        Generate and cache predictions for upcoming days
        
        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            print("Redis not available, cannot cache predictions")
            return False
            
        try:
            # Get latest business metrics
            latest_metrics = self._load_latest_data()
            print(f"Using latest business metrics: {latest_metrics}")
            
            # Generate predictions for upcoming days
            start_date = datetime.now()
            cached_count = 0
            
            for i in range(self.days_ahead):
                prediction_date = start_date + timedelta(days=i)
                date_str = prediction_date.strftime("%Y-%m-%d")
                
                # Make prediction
                prediction = self.predictor.predict(
                    date=prediction_date,
                    gmv=latest_metrics['gmv'],
                    users=latest_metrics['users'],
                    marketing_cost=latest_metrics['marketing_cost']
                )
                
                if prediction.get('fe_pods') is None or prediction.get('be_pods') is None:
                    print(f"Prediction failed for {date_str}")
                    continue
                
                # Prepare data for caching
                prediction_data = {
                    'fe_pods': prediction['fe_pods'],
                    'be_pods': prediction['be_pods'],
                    'date': date_str,
                    'cached': True,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Create cache key
                cache_key = f"prediction_forecast:{date_str}"
                
                # Cache prediction
                self.redis.setex(cache_key, self.cache_ttl, json.dumps(prediction_data))
                cached_count += 1
                
                print(f"Cached prediction for {date_str}: FE pods: {prediction['fe_pods']}, BE pods: {prediction['be_pods']}")
            
            print(f"Successfully cached predictions for {cached_count} days")
            return True
            
        except Exception as e:
            print(f"Error updating predictions: {str(e)}")
            return False


if __name__ == "__main__":
    import pandas as pd
    
    print("Starting prediction update job...")
    days_ahead = int(os.environ.get("PREDICTION_DAYS_AHEAD", 7))
    
    updater = PredictionUpdater(days_ahead=days_ahead)
    success = updater.update_predictions()
    
    sys.exit(0 if success else 1)
