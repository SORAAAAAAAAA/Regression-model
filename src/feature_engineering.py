import pandas as pd
import os 

def feature_engineering(df : pd.DataFrame, output_path: str) -> pd.DataFrame:
    
    # Check if the engineered data already exists, if so, read and return it
    if os.path.exists(output_path):
        print("Engineered data already exists. Skipping feature engineering.")
        return pd.read_csv(output_path)
    
    # Create a new feature: "age_of_car" based on the "year" column
    current_year = pd.Timestamp.now().year
    df["age_of_car"] = current_year - df["year"]
    
    # Create a new feature: "price_per_km" based on "price" and "km_driven"
    # Handle cases where "km_driven" is "N/A" or zero to avoid division by zero
    df["price_per_km"] = df.apply(lambda row: row["selling_price"] / row["km_driven"] if row["km_driven"] not in ["N/A", 0] else None, axis=1)
    
    # Create a new feature: "is_diesel" and "is_petrol" based on the "fuel" column
    df["is_diesel"] = df["fuel"].apply(lambda row: 1 if row == "diesel" else 0)
    
    # Create a new feature: "is_dealer" based on the "seller_type" column
    df["is_dealer"] = df["seller_type"].apply(lambda row: 1 if row == "dealer" else 0)  
    
    # Create a new feature: "is_automatic" based on the "transmission" column
    df["is_automatic"] = df["transmission"].apply(lambda row: 1 if row == "automatic" else 0)
    
    # Create a new feature: "owner_map" based on the "owner" column
    owner_map = {
        "first owner": 1,
        "second owner": 2,
        "third owner": 3,
        "fourth & above owner": 4,
    }
    df["owner_priority"] = df["owner"].map(owner_map).fillna(0).astype(int)
        
    # drop the original "name" and "year" columns as they are no longer needed
    df = df.drop(["name", "year", "fuel", "seller_type", "transmission", "owner"], axis=1)
    
    # Save the engineered dataframe to a new CSV file
    df.to_csv(output_path, index=False)
    
    return df