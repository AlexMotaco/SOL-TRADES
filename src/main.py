# src/main.py
import time
from strategies import arbitrage_strategy, trend_following_strategy, scalping_strategy

# Store historical prices for trend following and scalping
price_history = []

def main():
    token_pair = "SOL/USDT"
    
    while True:
        # Fetch current price from DEX for historical price data
        current_price = get_dex_price(token_pair)
        price_history.append(current_price)
        
        if len(price_history) > 20:
            # Keep price history manageable (only last 20 prices)
            price_history.pop(0)

        # Select strategy based on configuration or real-time conditions
        arbitrage_strategy(token_pair)
        trend_following_strategy(token_pair, price_history)
        scalping_strategy(token_pair, price_history)

        # Sleep for a bit before checking prices again
        time.sleep(10)  # Adjust this based on the speed of your trading strategy

if __name__ == "__main__":
    main()
