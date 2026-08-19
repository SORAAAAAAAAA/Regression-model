from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from data_transformation import transform_data
from feature_engineering import feature_engineering
import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict


def split_data(df: pd.DataFrame):
    # Split the data into features and target variable
    X = df.drop("selling_price", axis=1)
    y = df["selling_price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    return model

def predict_model(model, X_test):
    y_predict = model.predict(X_test)
    return y_predict

def evaluate_model(y_test, y_pred):
    # Evaluate the model on the test data
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return r2, mae, rmse

def main() -> None:
    # Define the path to the vehicle dataset
    vehicle_data_path = "src/data/vehicle-dataset/CAR DETAILS FROM CAR DEKHO.csv"
    
    # Transform the data and save it to a new CSV file
    print("Transforming data...")
    transformed_data_path = "src/data/processed_data/transformed/transformed_vehicle_data.csv"
    print("Successfully transformed data and saved to:", transformed_data_path)
    df = transform_data(vehicle_data_path, transformed_data_path)
    
    # Perform feature engineering on the transformed data and save it to a new CSV file
    engineered_data_path = "src/data/processed_data/engineered/engineered_vehicle_data.csv"
    print("Performing feature engineering...")
    df = feature_engineering(df, engineered_data_path)
    print("Successfully performed feature engineering and saved to:", engineered_data_path)
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test  = split_data(df)
    
    print("Training the model...")
    model = train_model(X_train, y_train)
    
    print("Making predictions on the test set...")
    y_pred = predict_model(model, X_test)
    
    print("Evaluating the model...")
    r2, mae, rmse = evaluate_model(y_test, y_pred)
    
    print("--- Model Performance Metrics ---")
    print(f"R-squared (R2): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}\n")
    
    
    
    

    
    

    
    
    
    
