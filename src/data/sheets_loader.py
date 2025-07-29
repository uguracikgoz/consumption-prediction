import os
import pandas as pd
from pathlib import Path
from typing import Optional
import gspread
from oauth2client.service_account import ServiceAccountCredentials


class GoogleSheetsLoader:
    def __init__(self, 
                credentials_file: str = None,
                sheet_key: str = None,
                worksheet_name: str = None):
        self.credentials_file = credentials_file or os.environ.get("GOOGLE_CREDENTIALS_FILE")
        self.sheet_key = sheet_key or os.environ.get("GOOGLE_SHEET_KEY")
        self.worksheet_name = worksheet_name or os.environ.get("GOOGLE_WORKSHEET_NAME", "Sheet1")
        
        if not self.credentials_file:
            raise ValueError("Google API credentials file not specified")
        
        if not self.sheet_key:
            raise ValueError("Google Sheet ID not specified")
    
    def _connect_to_sheets(self):
        try:
            print(f"Attempting to connect to Google Sheets using credentials: {self.credentials_file}")
            
            # Check if file exists
            if not os.path.exists(self.credentials_file):
                raise FileNotFoundError(f"Credentials file not found at: {self.credentials_file}")
                
            # Check if file is readable
            if not os.access(self.credentials_file, os.R_OK):
                raise PermissionError(f"Credentials file not readable at: {self.credentials_file}")
                
            print(f"Credentials file exists and is readable")
            
            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/drive"
            ]
            
            try:
                credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    self.credentials_file, scope
                )
                print(f"Successfully loaded credentials from JSON file")
            except Exception as json_error:
                print(f"Error loading JSON credentials: {str(json_error)}")
                raise
            
            try:
                client = gspread.authorize(credentials)
                print(f"Successfully authorized with gspread")
            except Exception as auth_error:
                print(f"Error authorizing with Google Sheets API: {str(auth_error)}")
                raise
                
            return client
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ConnectionError(f"Error connecting to Google Sheets API: {str(e)}")
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Processing data from Google Sheets")
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        if 'date' in df.columns:
            print("Converting date column from Google Sheets")
            print(f"Example date values: {df['date'].head(3).tolist()}")
            # Use DD/MM/YYYY format for dates coming from Google Sheets
            try:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
                print("Converted dates using format DD/MM/YYYY")
            except Exception as e:
                print(f"Error converting dates with DD/MM/YYYY format: {str(e)}")
                # Fallback: try with default parser
                try:
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True)  # Assume day is first
                    print("Converted dates using dayfirst=True")
                except Exception as e2:
                    print(f"Error with fallback date conversion: {str(e2)}")
                    raise
        
        # Handle empty values and clean numeric columns
        numeric_cols = ['gmv', 'marketing_cost', 'fe_pods', 'be_pods', 'users']
        for col in numeric_cols:
            if col in df.columns:
                print(f"Cleaning numeric column: {col}")
                # Convert empty strings to NaN
                df[col] = df[col].replace('', float('nan'))
                # If column is numeric with commas
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '').replace('nan', float('nan'))
                # Convert to float
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"Column {col} missing values: {df[col].isna().sum()} / {len(df)}")
        
        # Drop rows where all numeric values are missing
        original_len = len(df)
        df = df.dropna(subset=numeric_cols, how='all')
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows with all numeric values missing")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        try:
            # Connect to Google API
            client = self._connect_to_sheets()
            
            try:
                # Try to open the sheet by key
                print(f"Attempting to open sheet with key: {self.sheet_key}")
                sheet = client.open_by_key(self.sheet_key)
                print(f"Successfully opened sheet: {sheet.title}")
            except Exception as sheet_error:
                print(f"ERROR opening sheet by key: {type(sheet_error).__name__}: {str(sheet_error)}")
                print("Available spreadsheets for this service account:")
                try:
                    available_sheets = client.list_spreadsheet_files()
                    for s in available_sheets:
                        print(f"  - {s['name']} (ID: {s['id']})")
                except Exception as list_error:
                    print(f"Unable to list available sheets: {str(list_error)}")
                raise
            
            try:
                # Try to open the specific worksheet
                print(f"Attempting to open worksheet: {self.worksheet_name}")
                worksheet = sheet.worksheet(self.worksheet_name)
                print(f"Successfully opened worksheet: {self.worksheet_name}")
            except Exception as ws_error:
                print(f"ERROR opening worksheet {self.worksheet_name}: {type(ws_error).__name__}: {str(ws_error)}")
                print("Available worksheets:")
                try:
                    for ws in sheet.worksheets():
                        print(f"  - {ws.title}")
                except Exception as ws_list_error:
                    print(f"Unable to list worksheets: {str(ws_list_error)}")
                raise
            
            try:
                # Try to get the data
                print("Fetching records from worksheet...")
                data = worksheet.get_all_records()
                print(f"Successfully fetched {len(data)} records from worksheet")
                
                if not data:
                    print("WARNING: No data records found in the worksheet")
                else:
                    print(f"First record keys: {list(data[0].keys())}")
                
                df = pd.DataFrame(data)
                print(f"Created DataFrame with shape: {df.shape}")
            except Exception as data_error:
                print(f"ERROR fetching data from worksheet: {type(data_error).__name__}: {str(data_error)}")
                import traceback
                traceback.print_exc()
                raise
            
            # Apply data cleaning and transformation
            return self._process_data(df)
            
            
        except Exception as e:
            raise Exception(f"Error loading data from Google Sheets: {str(e)}")


if __name__ == "__main__":
    try:
        # First check if environment variables are set
        if (os.environ.get("GOOGLE_CREDENTIALS_FILE") and 
            os.environ.get("GOOGLE_SHEET_KEY")):
            
            loader = GoogleSheetsLoader()
            df = loader.load_data()
            print(f"Successfully loaded data from Google Sheets")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"First few rows:")
            print(df.head())
        else:
            print("Google Sheets environment variables not set. Skipping test.")
    except Exception as e:
        print(f"Error in Google Sheets loader test: {str(e)}")
