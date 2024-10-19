# src/strategies.py
import logging
from exchanges import get_dex_price, get_cex_price
from config import ARBITRAGE_THRESHOLD

# Arbitrage Strategy
def arbitrage_strategy(token_pair):
    dex_price = get_dex_price(token_pair)
    cex_price = get_cex_price(token_pair)

    if dex_price > cex_price * ARBITRAGE_THRESHOLD:
        logging.info(f"Arbitrage opportunity! Buy on FTX: {cex_price}, Sell on Jupiter: {dex_price}")
        # Add buy/sell logic here
    else:
        logging.info("No arbitrage opportunity found.")

# Trend Following Strategy
def trend_following_strategy(token_pair, price_history):
    # Calculate moving averages
    short_term_avg = sum(price_history[-5:]) / 5
    long_term_avg = sum(price_history[-20:]) / 20

    if short_term_avg > long_term_avg:
        logging.info(f"Uptrend detected for {token_pair}. Consider buying.")
        # Add buy logic here
    else:
        logging.info(f"Downtrend detected for {token_pair}. Consider selling.")
        # Add sell logic here

# Scalping Strategy
def scalping_strategy(token_pair, price_history):
    # Example scalping strategy using price volatility
    last_price = price_history[-1]
    price_change = abs(last_price - price_history[-2])

    if price_change > 0.02 * last_price:  # 2% volatility threshold
        logging.info(f"Scalping opportunity detected on {token_pair}.")
        # Add quick buy/sell logic here
    else:
        logging.info("No scalping opportunity at the moment.")
