import requests
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from configparser import ConfigParser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ApiService:
    def __init__(self):
        self.config = self.load_config()
        self.api_url = self.config['API']['url']
        self.api_key = self.config['API']['key']
        self.model = None  # Placeholder for AI model
        self.historical_data = []  # Store historical price data for AI

    def load_config(self):
        config = ConfigParser()
        config.read('config.ini')
        return config

    def fetch_price(self, symbol):
        try:
            response = requests.get(f"{self.api_url}/price?symbol={symbol}", headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            price_data = response.json()
            logging.info(f"Fetched price data: {price_data}")
            return price_data['price']
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            return None

    def place_order(self, symbol, amount, order_type='buy'):
        try:
            order_data = {'symbol': symbol, 'amount': amount, 'type': order_type}
            response = requests.post(f"{self.api_url}/order", json=order_data, headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            order_response = response.json()
            logging.info(f"Order placed: {order_response}")
            return order_response
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            return None

    def get_historical_data(self, symbol, days=30):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            response = requests.get(f"{self.api_url}/historical?symbol={symbol}&start={start_date}&end={end_date}", headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            historical_prices = response.json()['prices']
            self.historical_data = historical_prices  # Save historical data for AI
            logging.info("Fetched historical price data.")
            return historical_prices
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            return None
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            return None

    def train_model(self):
        # Train AI model with historical data
        if not self.historical_data:
            logging.warning("No historical data available to train the model.")
            return
        
        prices = np.array([data['price'] for data in self.historical_data]).reshape(-1, 1)
        days = np.array(range(len(prices))).reshape(-1, 1)  # Simple day index
        
        self.model = LinearRegression()
        self.model.fit(days, prices)
        logging.info("AI model trained.")

    def predict_price(self, current_day):
        if self.model is None:
            logging.warning("Model not trained. Unable to predict price.")
            return None
        
        predicted_price = self.model.predict(np.array([[current_day + 1]]))  # Predict the next day's price
        return predicted_price[0][0]

    def dynamic_risk_management(self, current_price, entry_price, stop_loss_pct=0.05, take_profit_pct=0.10):
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct)
        
        if current_price <= stop_loss_price:
            logging.info("Stop loss triggered.")
            return 'sell'
        elif current_price >= take_profit_price:
            logging.info("Take profit triggered.")
            return 'sell'
        
        return 'hold'

if __name__ == "__main__":
    api_service = ApiService()
    symbol = 'SOL'
    api_service.get_historical_data(symbol)
    api_service.train_model()
    
    current_day = len(api_service.historical_data) - 1
    predicted_price = api_service.predict_price(current_day)
    
    logging.info(f"Predicted price for next day: {predicted_price}")

    # Example order placement
    current_price = api_service.fetch_price(symbol)
    if current_price:
        action = api_service.dynamic_risk_management(current_price, predicted_price)
        if action == 'sell':
            api_service.place_order(symbol, amount=1, order_type='sell')
