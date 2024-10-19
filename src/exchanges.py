# src/exchanges.py
import requests
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from config import JUPITER_API_URL, FTX_API_URL, STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, TRADING_AMOUNT

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def place_ftx_buy_order(token_pair, amount):
    try:
        # Example API call to create a market buy order
        order = {
            'average': 100,  # This is a placeholder; you should get this from the API response.
            'token_pair': token_pair,
            'amount': amount,
        }
        logging.info(f"Buy order placed on FTX: {order}")

        # Implement stop-loss and take-profit
        stop_loss_price = order['average'] * (1 - STOP_LOSS_PERCENTAGE)
        take_profit_price = order['average'] * (1 + TAKE_PROFIT_PERCENTAGE)

        logging.info(f"Stop-loss set at: {stop_loss_price}, Take-profit set at: {take_profit_price}")

        return order
    except Exception as e:
        logging.error(f"Error placing FTX buy order: {e}")
