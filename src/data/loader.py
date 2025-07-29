import os
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Union, Literal

try:
    from .sheets_loader import GoogleSheetsLoader
    SHEETS_SUPPORT = True
except ImportError:
    SHEETS_SUPPORT = False


class DataLoader:
    def __init__(self, 
                data_path: str = None, 
                source_type: Literal['csv', 'sheets'] = 'sheets',
                sheets_config: Dict = None):
        self.source_type = source_type
        
        if data_path is None:
            possible_paths = [
                os.environ.get('DATA_FILE_PATH'),
                "/app/data.csv",
                "data.csv",
                str(Path(__file__).parent.parent.parent.parent / "data.csv")
            ]
            
            possible_paths = [p for p in possible_paths if p]
            
            # Use the first path that exists, or default to the last one
            for path in possible_paths:
                if os.path.exists(path):
                    self.data_path = path
                    break
            else:
                # If no file exists, use the first path as default
                self.data_path = possible_paths[0] if possible_paths else "data.csv"
                
            print(f"Using data file path: {self.data_path}")
        else:
            self.data_path = data_path
            
        # Google Sheets setup
        self.sheets_config = sheets_config or {}
        
        # Verify Google Sheets is available if selected
        if source_type == 'sheets' and not SHEETS_SUPPORT:
            raise ImportError("Google Sheets support requires gspread and oauth2client packages. "
                            "Please install them with: pip install gspread oauth2client")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from the configured data source (CSV or Google Sheets)
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            if self.source_type == 'csv':
                return self._load_from_csv()
            elif self.source_type == 'sheets':
                return self._load_from_sheets()
            else:
                raise ValueError(f"Unsupported source type: {self.source_type}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _load_from_csv(self) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Fix column names to be snake_case
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Convert date column to datetime with explicit format (DD/MM/YYYY)
            try:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
                print(f"Converted dates using format DD/MM/YYYY")
            except Exception as e:
                print(f"Failed to parse dates with DD/MM/YYYY format: {e}")
                try:
                    # Fallback to automatic parsing
                    df['date'] = pd.to_datetime(df['date'])
                    print(f"Converted dates using automatic parsing")
                except Exception as e:
                    print(f"Failed to parse date column: {e}")
                    raise
            
            # Replace commas in numeric columns and convert to float
            for col in ['gmv', 'marketing_cost']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading data from CSV: {str(e)}")
    
    def _load_from_sheets(self) -> pd.DataFrame:
        """
        Load data from Google Sheets
        
        Returns:
            DataFrame containing the loaded data
        """
        try:
            # Check for required configuration
            credentials_file = self.sheets_config.get('credentials_file') or os.environ.get("GOOGLE_CREDENTIALS_FILE")
            sheet_key = self.sheets_config.get('sheet_key') or os.environ.get("GOOGLE_SHEET_KEY")
            worksheet_name = self.sheets_config.get('worksheet_name') or os.environ.get("GOOGLE_WORKSHEET_NAME", "Sheet1")
            
            if not credentials_file or not sheet_key:
                raise ValueError("Missing Google Sheets configuration. Set credentials_file and sheet_key.")
            
            # Initialize the Google Sheets loader
            sheets_loader = GoogleSheetsLoader(
                credentials_file=credentials_file,
                sheet_key=sheet_key,
                worksheet_name=worksheet_name
            )
            
            # Load data from sheets
            return sheets_loader.load_data()
        except Exception as e:
            raise Exception(f"Error loading data from Google Sheets: {str(e)}")

    
    def preprocess_data(self, df: pd.DataFrame, 
                       target_columns: list = ['fe_pods', 'be_pods']) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Preprocess data for machine learning
        
        Args:
            df: DataFrame containing the raw data
            target_columns: List of target columns to predict
            
        Returns:
            Tuple of (features_df, targets_df) where targets_df may be None if targets are not available
        """
        try:
            # Create copy to avoid modifying the original
            processed_df = df.copy()
            
            # Extract date features
            processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
            processed_df['week_of_year'] = processed_df['date'].dt.isocalendar().week
            processed_df['month'] = processed_df['date'].dt.month
            processed_df['day_of_month'] = processed_df['date'].dt.day
            processed_df['is_weekend'] = processed_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Set date as index but keep it as a feature too
            processed_df.set_index('date', inplace=True, drop=False)
            
            # Prepare features and targets
            features = processed_df.drop(columns=['date'] + [col for col in target_columns if col in processed_df.columns])
            
            # Check if targets exist in the data
            targets = None
            if all(col in processed_df.columns for col in target_columns):
                if not processed_df[target_columns].isna().all().all():  # Check if there's at least some target data
                    targets = processed_df[target_columns]
            
            return features, targets
        except Exception as e:
            raise Exception(f"Error preprocessing data: {str(e)}")


if __name__ == "__main__":
    # Test loading and preprocessing
    loader = DataLoader()
    df = loader.load_data()
    features, targets = loader.preprocess_data(df)
    
    print(f"Loaded data shape: {df.shape}")
    print(f"Features shape: {features.shape}")
    if targets is not None:
        print(f"Targets shape: {targets.shape}")
    else:
        print("No target data found")
