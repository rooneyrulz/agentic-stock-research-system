# NSE Stock Market Parameters
NSE_TRADING_HOURS = {
    "market_open": "09:15",
    "market_close": "15:30",
    "pre_open_start": "09:00",
    "pre_open_end": "09:08",
}

# Stock Selection Criteria
STOCK_SELECTION_CRITERIA = {
    "min_price": 50.0,  # Minimum stock price in ₹
    "min_market_cap_cr": 5000.0,  # Minimum market cap in crores
    "min_daily_volume_cr": 50.0,  # Minimum daily volume in crores
    "min_stocks": 2,
    "max_stocks": 3,
}

# Technical Indicator Thresholds
TECHNICAL_THRESHOLDS = {
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "volume_surge_multiplier": 1.5,  # 50% above average
    "strong_momentum_pct": 5.0,  # 5% price change
    "weak_momentum_pct": 1.0,
}

# Risk Management Parameters
RISK_MANAGEMENT = {
    "max_stop_loss_pct": 5.0,  # Maximum 5% stop loss
    "min_risk_reward_ratio": 1.5,  # Minimum 1:1.5 ratio
    "position_size_pct": 5.0,  # Maximum 5% of portfolio per stock
}
