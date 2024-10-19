# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Jupiter and FTX API configuration
JUPITER_API_URL = os.getenv("JUPITER_API_URL")
FTX_API_URL = os.getenv("FTX_API_URL")

# Trading configuration
TRADING_AMOUNT = float(os.getenv("TRADING_AMOUNT", 1.0))  # Default trading amount
STOP_LOSS_PERCENTAGE = float(os.getenv("STOP_LOSS_PERCENTAGE", 0.02))  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = float(os.getenv("TAKE_PROFIT_PERCENTAGE", 0.03))  # 3% take profit
