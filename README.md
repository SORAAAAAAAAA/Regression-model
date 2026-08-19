# Vehicle Price Regression Model

This project implements a complete machine learning pipeline to predict the selling price of used cars using a Linear Regression model. It fetches a dataset from Kaggle, performs data cleaning and feature engineering, and evaluates the trained model on test data.

## Project Structure

```text
regression_model/
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # Project documentation
└── src/
    ├── data/
    │   └── __init__.py    # Script to download dataset from Kaggle
    ├── data_transformation.py  # Data cleaning and preprocessing
    ├── feature_engineering.py  # Feature extraction and encoding
    └── regression_model/
        └── __init__.py    # Main training and evaluation pipeline
```

## Prerequisites

- Python 3.14 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended, as the project uses `uv_build` backend)

## Installation

You can install the dependencies and the project in a virtual environment. Since the project uses `uv`, you can run:

```bash
# Create a virtual environment and sync dependencies
uv sync
```
Alternatively, if you're using `pip`:
```bash
pip install -e .
```

## Usage

This project defines two main executable scripts which can be run after installation.

### 1. Download the Dataset

Before running the model, you must download the dataset. The data is fetched from the Kaggle dataset `nehalbirla/vehicle-dataset-from-cardekho`.

```bash
# Using uv run
uv run download-data

# Or directly if the environment is activated
download-data
```

This will download and extract the dataset into `src/data/vehicle-dataset/`.

### 2. Run the Regression Model Pipeline

To execute the entire machine learning pipeline:

```bash
# Using uv run
uv run regression-model

# Or directly if the environment is activated
regression-model
```

This will sequentially perform the following steps:
1. **Data Transformation**: Cleans column names, converts categorical values to lowercase, and handles missing values. The transformed data is saved to `src/data/processed_data/transformed/`.
2. **Feature Engineering**: Derives new features such as `age_of_car` and `price_per_km`, creates binary indicators for fuel types and transmission, and maps owner history to numerical priorities. The engineered data is saved to `src/data/processed_data/engineered/`.
3. **Model Training**: Splits the data into training (80%) and testing (20%) sets, and trains a scikit-learn `LinearRegression` model.
4. **Evaluation**: Predicts prices on the test set and outputs R-squared (R2), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) metrics.

## Dependencies

- `kagglehub` (>=1.0.2)
- `pandas` (>=3.0.5)
- `scikit-learn` (>=1.9.0)
