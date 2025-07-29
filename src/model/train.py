import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import DataLoader


class ModelTrainer:
    def __init__(self, 
                 model_dir: str = None,
                 test_size: float = 0.2,
                 random_state: int = 42):
        if model_dir is None:
            self.model_dir = str(Path(__file__).parent / "saved_models")
        else:
            self.model_dir = model_dir
            
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
    
    def split_data(self, 
                  features: pd.DataFrame, 
                  targets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            features: DataFrame of feature variables
            targets: DataFrame of target variables
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, 
                test_size=self.test_size,
                random_state=self.random_state,
                shuffle=False  # Time series data shouldn't be shuffled
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise Exception(f"Error splitting data: {str(e)}")
    
    def train_models(self, 
                     features: pd.DataFrame, 
                     targets: pd.DataFrame,
                     model_params: Dict = None) -> Dict:
        """
        Train models for each target column using an ensemble approach
        
        Args:
            features: DataFrame of feature variables
            targets: DataFrame of target variables
            model_params: Parameters for models
            
        Returns:
            Dictionary of trained models for each target
        """
        # Analyze feature correlations with targets
        print("\nAnalyzing feature correlations with target variables:")
        correlation_data = pd.concat([features, targets], axis=1)
        for column in targets.columns:
            correlations = correlation_data.corr()[column].sort_values(ascending=False)
            print(f"\nCorrelations with {column}:")
            print(correlations.drop(targets.columns))
        
        # Create improved feature set with interaction terms and user-focused features
        enhanced_features = features.copy()
        
        # Handle any potential division by zero
        enhanced_features['gmv_per_user'] = enhanced_features['gmv'].div(enhanced_features['users'].replace(0, np.nan)).fillna(0)
        enhanced_features['marketing_per_user'] = enhanced_features['marketing_cost'].div(enhanced_features['users'].replace(0, np.nan)).fillna(0)
        
        # Add non-linear and logarithmic transformations of users to capture scaling effects
        enhanced_features['user_sqrt'] = np.sqrt(enhanced_features['users'])  # Square root to reduce skew
        enhanced_features['user_log'] = np.log1p(enhanced_features['users'])  # Log transform to reduce skew
        enhanced_features['user_squared'] = np.power(enhanced_features['users'], 2)  # Squared to capture non-linear growth
        
        # Add combined metrics that might better correlate with pod requirements
        enhanced_features['user_gmv_product'] = enhanced_features['users'] * enhanced_features['gmv']  # User-GMV interaction
        enhanced_features['user_to_gmv_ratio'] = enhanced_features['users'].div(enhanced_features['gmv'].replace(0, np.nan)).fillna(0)  # Users per GMV
        
        # Traffic-based features (assuming GMV and users represent traffic patterns)
        enhanced_features['traffic_index'] = enhanced_features['users'] * enhanced_features['gmv_per_user']  # Weighted user traffic
        enhanced_features['peak_factor'] = enhanced_features['users'] * (1 + enhanced_features['marketing_cost'] / (enhanced_features['gmv'] + 1))  # Marketing-driven peaks
        
        # Time-based interaction features
        if 'day_of_week' in enhanced_features.columns:
            # Weekday vs weekend effect on user load
            enhanced_features['weekend_users'] = enhanced_features['users'] * enhanced_features['is_weekend']
            # Day of week interaction with users (captures weekly patterns)
            enhanced_features['day_user_interaction'] = enhanced_features['users'] * enhanced_features['day_of_week']
        
        # Normalize key features to help the model
        for col in ['users', 'gmv', 'marketing_cost']:
            if col in enhanced_features.columns:
                enhanced_features[f'{col}_normalized'] = (enhanced_features[col] - enhanced_features[col].mean()) / enhanced_features[col].std()
        
        # Set default model parameters if none provided
        if model_params is None:
            # Use more conservative hyperparameters for small dataset (30 rows) to avoid overfitting
            model_params = {
                'n_estimators': 100,  # Reduced number of trees to avoid overfitting
                'max_depth': 3,       # Shallow trees to prevent memorizing the training data
                'learning_rate': 0.05, # Slower learning rate for better generalization
                'min_samples_split': 2, # Minimum samples required to split a node
                'min_samples_leaf': 1,  # Minimum samples required at a leaf node
                'subsample': 0.8,      # Use 80% of samples for each tree to reduce variance
                'random_state': self.random_state
            }
            
        try:
            # Train a separate model for each target
            models = {}
            results = {}
            
            for column in targets.columns:
                print(f"\nTraining model for {column}...")
                
                # Get single target column
                target = targets[column]
                
                # Split data
                X_train, X_test, y_train, y_test = self.split_data(enhanced_features, target)
                
                # Create and train model
                from sklearn.model_selection import GridSearchCV
                
                # Find best parameters using grid search
                print(f"Finding optimal parameters for {column}...")
                param_grid = {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
                
                # Use GradientBoostingRegressor which often handles small datasets better
                model = GradientBoostingRegressor(random_state=self.random_state)
                grid_search = GridSearchCV(model, param_grid, cv=min(5, len(X_train)), n_jobs=-1, scoring='neg_mean_absolute_error')
                grid_search.fit(X_train, y_train)
                
                print(f"Best parameters for {column}: {grid_search.best_params_}")
                model = grid_search.best_estimator_
                
                # Evaluate model
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Store metrics
                results[column] = {
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse
                }
                
                # Store model
                models[column] = model
                
                print(f"Model for {column} - Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': enhanced_features.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"Top 5 features for {column}:")
                print(feature_importance.head(5))
            
            # Save evaluation results
            results_df = pd.DataFrame({
                'target': list(results.keys()),
                'train_mae': [results[k]['train_mae'] for k in results],
                'test_mae': [results[k]['test_mae'] for k in results],
                'train_rmse': [results[k]['train_rmse'] for k in results],
                'test_rmse': [results[k]['test_rmse'] for k in results]
            })
            
            # Store models for later use
            self.models = models
            
            return models, results_df
            
        except Exception as e:
            raise Exception(f"Error training models: {str(e)}")
    
    def save_models(self, models: Dict = None) -> None:
        """
        Save trained models to disk
        
        Args:
            models: Dictionary of models to save. If None, uses self.models.
        """
        if models is None:
            models = self.models
            
        try:
            for target, model in models.items():
                model_path = os.path.join(self.model_dir, f"{target}_model.joblib")
                joblib.dump(model, model_path)
                print(f"Model for {target} saved to {model_path}")
        except Exception as e:
            raise Exception(f"Error saving models: {str(e)}")
    
    def load_models(self, target_columns: List[str] = None) -> Dict:
        """
        Load trained models from disk
        
        Args:
            target_columns: List of target columns to load models for
            
        Returns:
            Dictionary of loaded models
        """
        if target_columns is None:
            target_columns = ['fe_pods', 'be_pods']
            
        try:
            loaded_models = {}
            for target in target_columns:
                model_path = os.path.join(self.model_dir, f"{target}_model.joblib")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    loaded_models[target] = model
                    print(f"Loaded model for {target} from {model_path}")
                else:
                    print(f"No saved model found for {target} at {model_path}")
            
            self.models = loaded_models
            return loaded_models
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")


if __name__ == "__main__":
    # Test training models
    loader = DataLoader()
    df = loader.load_data()
    
    # For the purpose of this example, let's generate some fake pod data
    # since the original data has empty fe_pods and be_pods columns
    if df['fe_pods'].isna().all() and df['be_pods'].isna().all():
        print("Generating fake pod data for demonstration...")
        # Generate synthetic data based on GMV and users
        df['fe_pods'] = (df['gmv'] / 1_000_000 * 0.5 + df['users'] / 10000 * 0.3).round().astype(int)
        df['be_pods'] = (df['gmv'] / 1_000_000 * 0.3 + df['users'] / 10000 * 0.7).round().astype(int)
        
        # Add some noise
        np.random.seed(42)
        df['fe_pods'] += np.random.randint(-2, 3, size=len(df))
        df['be_pods'] += np.random.randint(-1, 2, size=len(df))
        
        # Ensure minimum values
        df['fe_pods'] = df['fe_pods'].clip(lower=1)
        df['be_pods'] = df['be_pods'].clip(lower=1)
    
    # Preprocess data
    features, targets = loader.preprocess_data(df)
    
    # Train models
    trainer = ModelTrainer()
    models, results = trainer.train_models(features, targets)
    
    # Save models
    trainer.save_models(models)
    
    print("\nModel Training Results:")
    print(results)
