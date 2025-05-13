# AI Trading Bot

An automated trading bot using machine learning and sentiment analysis for options trading on the Alpaca platform.

## Features

- Machine learning-based trading signals
- Market sentiment analysis
- Options trading strategy
- Real-time dashboard
- Backtesting capabilities
- Risk management

## Prerequisites

- Python 3.8+
- Alpaca Trading API credentials
- TA-Lib (Technical Analysis Library)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd trading
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Alpaca credentials:
```
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Use paper trading for testing
```

## Usage

1. Start the dashboard:
```bash
python -m app.dashboard
```

2. Access the dashboard at `http://localhost:8050`

3. Enter a stock symbol to analyze trading signals, market sentiment, and backtest results.

## Project Structure

```
trading/
├── app/
│   ├── ml/
│   │   ├── models.py
│   │   ├── backtesting.py
│   │   └── sentiment.py
│   ├── trading/
│   │   └── options_strategy.py
│   └── dashboard.py
├── requirements.txt
├── .env
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS. 