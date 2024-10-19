import requests
import logging
import configparser
import time
import numpy as np
import pandas as pd  # For data handling
from sklearn.linear_model import LinearRegression  # Example ML model
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Constants
API_URL = config.get('API', 'api_url', fallback='https://api.solana.com')
TOKEN = config.get('API', 'token', fallback='your_api_token')
TRADING_PAIR = config.get('Trading', 'trading_pair', fallback='SOL/USDT')
POSITION_SIZE = config.getfloat('Trading', 'position_size', fallback=0.01)  # Percentage of balance to use
STOP_LOSS_PERCENTAGE = config.getfloat('Trading', 'stop_loss_percentage', fallback=0.02)  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = config.getfloat('Trading', 'take_profit_percentage', fallback=0.05)  # 5% take profit

# Initialize a linear regression model for prediction (example AI model)
model = LinearRegression()

# Function to get historical price data
def get_historical_data():
    try:
        response = requests.get(f"{API_URL}/historical/prices/{TRADING_PAIR}", headers={"Authorization": f"Bearer {TOKEN}"})
        response.raise_for_status()
        historical_data = response.json()
        logging.info("Historical data retrieved.")
        return pd.DataFrame(historical_data)  # Assuming response is convertible to DataFrame
    except Exception as e:
        logging.error("Error retrieving historical data: %s", e)
        return pd.DataFrame()  # Return empty DataFrame on error

# Function to train the AI model on historical data
def train_model(historical_data):
    try:
        X = historical_data[['feature1', 'feature2']]  # Replace with actual features
        y = historical_data['target']  # Replace with actual target
        model.fit(X, y)
        logging.info("Model trained successfully.")
    except Exception as e:
        logging.error("Error training model: %s", e)

# Function to predict future price movement using the AI model
def predict_price(data):
    try:
        prediction = model.predict(data[['feature1', 'feature2']])  # Use the same features as training
        return prediction[-1]  # Return the last prediction
    except Exception as e:
        logging.error("Error predicting price: %s", e)
        return None

# Function to get account balance
def get_balance():
    try:
        response = requests.get(f"{API_URL}/account/balance", headers={"Authorization": f"Bearer {TOKEN}"})
        response.raise_for_status()
        balance_data = response.json()
        balance = balance_data['balance']
        logging.info("Account balance retrieved: %s", balance)
        return balance
    except Exception as e:
        logging.error("Error retrieving account balance: %s", e)
        return 0  # Return 0 on error to prevent trades

# Function to execute a trade
def execute_trade(action, amount):
    try:
        trade_data = {
            'symbol': TRADING_PAIR,
            'action': action,
            'amount': amount
        }
        response = requests.post(f"{API_URL}/trade", json=trade_data, headers={"Authorization": f"Bearer {TOKEN}"})
        response.raise_for_status()
        logging.info("Trade executed: %s %s %s", action, amount, TRADING_PAIR)
    except Exception as e:
        logging.error("Error executing trade: %s", e)
        time.sleep(5)  # Wait and retry for transient errors
        execute_trade(action, amount)  # Retry trade execution

# Function to determine trade signal using AI predictions
def determine_signal(price_data):
    try:
        predicted_price = predict_price(price_data)
        if predicted_price is not None:
            if predicted_price < price_data['latest_price']:
                return "buy"
            elif predicted_price > price_data['latest_price']:
                return "sell"
        return "hold"  # No action
    except Exception as e:
        logging.error("Error determining signal: %s", e)
        return "hold"  # Default to holding position on error

# Function to implement stop loss and take profit logic
def manage_position(current_price, entry_price):
    stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENTAGE)
    take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENTAGE)
    
    if current_price <= stop_loss_price:
        logging.info("Stop loss triggered at %s. Exiting position.", current_price)
        return "sell"
    elif current_price >= take_profit_price:
        logging.info("Take profit triggered at %s. Exiting position.", current_price)
        return "sell"
    return "hold"  # No action

# Main trading loop
def trading_loop():
    entry_price = None  # Initialize entry price
    historical_data = get_historical_data()
    train_model(historical_data)  # Train the model at the start

    while True:
        try:
            # Simulate fetching price data
            price_data = {
                'latest_price': 50,  # Replace with actual API call
                'feature1': 1,  # Replace with actual feature
                'feature2': 2   # Replace with actual feature
            }

            action = determine_signal(price_data)
            balance = get_balance()
            amount_to_trade = balance * POSITION_SIZE

            if action == "buy" and entry_price is None:  # Only buy if not already in a position
                execute_trade("buy", amount_to_trade)
                entry_price = price_data['latest_price']  # Set entry price
            elif action == "sell" and entry_price is not None:  # Only sell if in a position
                execute_trade("sell", amount_to_trade)
                entry_price = None  # Reset entry price after selling
            elif entry_price is not None:  # Check for stop loss or take profit conditions
                manage_position(price_data['latest_price'], entry_price)

            time.sleep(60)  # Pause before next trading action

        except Exception as e:
            logging.error("Error in trading loop: %s", e)
            time.sleep(10)  # Wait before retrying the trading loop

# Entry point
if __name__ == "__main__":
    trading_loop()
