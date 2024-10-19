import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt
import configparser
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
DATA_PATH = config.get('Paths', 'data_path', fallback='data/market_data.csv')
MODEL_PATH = config.get('Paths', 'model_path', fallback='model/trading_model.pkl')
THRESHOLD = config.getfloat('Trading', 'threshold', fallback=0.5)

# Data Preprocessing Function
def preprocess_data(df):
    logging.info("Starting data preprocessing...")
    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['price', 'volume']])  # Adjust as per your feature set
    df[['price', 'volume']] = scaled_features

    return df

# Model Training Function
def train_model(X_train, y_train):
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    logging.info("Best parameters found: %s", grid_search.best_params_)
    
    # Save the model
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)
    logging.info("Model saved to %s", MODEL_PATH)
    
    return grid_search.best_estimator_

# Model Evaluation Function
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logging.info("Model accuracy: %.2f%%", accuracy * 100)
    logging.info("Classification report:\n%s", classification_report(y_test, predictions))
    return predictions

# Backtesting Function
def backtest_strategy(df):
    logging.info("Starting backtesting...")
    results = []
    
    for threshold in [0.4, 0.5, 0.6]:
        df['signal'] = np.where(df['predictions'] > threshold, 1, 0)
        df['strategy_return'] = df['returns'] * df['signal'].shift(1)
        cumulative_return = (1 + df['strategy_return']).cumprod() - 1
        results.append(cumulative_return)
    
    return results

# Visualization Function
def visualize_results(results):
    plt.figure(figsize=(10, 6))
    for result in results:
        plt.plot(result, label=f'Strategy Threshold {result.name}')
    
    plt.title('Cumulative Returns')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

# Load Data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info("Data loaded successfully from %s", DATA_PATH)
        return df
    except Exception as e:
        logging.error("Error loading data: %s", e)
        return pd.DataFrame()  # Return empty DataFrame on error

# Main Execution
if __name__ == "__main__":
    market_data = load_data()
    
    if not market_data.empty:
        market_data = preprocess_data(market_data)

        # Feature engineering (e.g., returns calculation)
        market_data['returns'] = market_data['price'].pct_change()
        market_data.dropna(inplace=True)

        # Train/Test Split
        X = market_data[['price', 'volume']]  # Adjust as per your feature set
        y = market_data['signal']  # Placeholder for actual signal

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        predictions = evaluate_model(model, X_test, y_test)

        # Backtest the strategy
        backtest_results = backtest_strategy(market_data)

        # Visualize the results
        visualize_results(backtest_results)

    else:
        logging.error("No data to process. Exiting.")
