import pandas as pd
import os

def transform_data(input_path: str, output_path: str) -> pd.DataFrame:
    
    if os.path.exists(output_path):
        print("Transformed data already exists. Skipping transformation.")
        return pd.read_csv(output_path)
    
    df = pd.read_csv(input_path)
    
    # Clean column names: strip whitespace and convert to lowercase
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Clean rows:  convert to lowercase
    df["name"] = df["name"].str.lower()
    df["seller_type"] = df["seller_type"].str.lower()
    df["fuel"] = df["fuel"].str.lower()
    df["transmission"] = df["transmission"].str.lower()
    df["owner"] = df["owner"].str.lower()
    
    # Fill missing values in the "km_driven" column with "N/A"
    df["km_driven"] = df["km_driven"].fillna("N/A")
    
    # Return the dataframe with cleaned data
    df.to_csv(output_path, index=False)
    
    return df