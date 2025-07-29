#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from data.loader import DataLoader
from model.train import ModelTrainer


def init_models():
    print("Initializing pod prediction models...")
    
    try:
        # First try loading from Google Sheets if credentials are available
        source_type = 'csv'
        if os.environ.get("GOOGLE_CREDENTIALS_FILE") and os.environ.get("GOOGLE_SHEET_KEY"):
            print(f"Google Sheets credentials found at {os.environ.get('GOOGLE_CREDENTIALS_FILE')}")
            print(f"Google Sheet Key: {os.environ.get('GOOGLE_SHEET_KEY')}")
            print(f"Google Worksheet Name: {os.environ.get('GOOGLE_WORKSHEET_NAME', 'Sheet1')}")
            
            # Verify credentials file exists
            if os.path.exists(os.environ.get("GOOGLE_CREDENTIALS_FILE")):
                print(f"Credentials file exists and is readable")
                # Set source type to sheets
                source_type = 'sheets'
                print("Using Google Sheets as data source")
            else:
                print(f"WARNING: Credentials file does not exist at {os.environ.get('GOOGLE_CREDENTIALS_FILE')}")
                print(f"Falling back to CSV source")
            
        # Initialize data loader with appropriate source
        loader = DataLoader(source_type=source_type)
        print(f"Data loader initialized with source_type: {source_type}")
        
        # Load data
        df = loader.load_data()
        print(f"Loaded data with shape: {df.shape} from source: {source_type}")
        
        print(f"Missing values in fe_pods: {df['fe_pods'].isna().sum()} / {len(df)}")
        print(f"Missing values in be_pods: {df['be_pods'].isna().sum()} / {len(df)}")
        
        df_clean = df.dropna(subset=['fe_pods', 'be_pods'])
        
        if len(df_clean) == 0:
            print("No rows with complete pod data found. Cannot continue.")
            return False
        
        print(f"Using {len(df_clean)} rows with real pod data for training (dropped {len(df) - len(df_clean)} rows with NaN values)")
        df = df_clean
        
        features, targets = loader.preprocess_data(df)
        print(f"Processed features with shape: {features.shape}")
        
        if targets is None:
            raise ValueError("No target data available for training.")
            
        print(f"Processed targets with shape: {targets.shape}")
        
        # Train models
        trainer = ModelTrainer()
        models, results = trainer.train_models(features, targets)
        
        # Save models
        trainer.save_models(models)
        
        print("\nModel training results:")
        print(results)
        print("\nModels have been successfully initialized!")
        
        return True
        
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return False


if __name__ == "__main__":
    success = init_models()
    sys.exit(0 if success else 1)
