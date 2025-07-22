# AI-Driven Simple Trend/Reversal Backtester

![Example Plot](https://via.placeholder.com/600x300?text=Backtest+Performance+Plot) A Python-based project that demonstrates how to build a simple algorithmic trading backtester driven by a Machine Learning model. This project aims to predict short-term stock price movements (up/down) using technical indicators and uses these predictions to simulate a basic trading strategy.

**Disclaimer:** This project is for **educational purposes and skill demonstration only**. It is not intended for live trading and does not account for real-world complexities such as slippage, transaction costs, market microstructure, or regulatory compliance. Trading financial instruments carries significant risk, and you could lose money.

## Features

* **Historical Data Acquisition:** Fetches daily stock/crypto data using `yfinance`.
* **Feature Engineering:** Calculates common technical indicators (e.g., SMA, RSI, MACD, Bollinger Bands, ROC) from raw price data.
* **Target Variable Creation:** Generates a binary target (1 for price up, 0 for price down/flat) for prediction.
* **Machine Learning Model:** Trains a `RandomForestClassifier` (or other scikit-learn model) to predict future price direction.
* **Backtesting Engine:** Simulates a basic long-only trading strategy based on the AI model's buy signals.
* **Performance Analysis:** Calculates key metrics like Total Return, Annualized Return, Max Drawdown, and Sharpe Ratio.
* **Visualization:** Plots portfolio equity curve and buy/sell signals on the price chart.

## Getting Started

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/varaprasad-jada/ai-driven-backtester.git](https://github.com/varaprasad-jada/ai-driven-backtester.git)
    cd ai-driven-backtester
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure `main.py`:**
    Open `main.py` and modify the `Configuration` section:
    * `STOCK_TICKER`: The stock symbol you want to backtest (e.g., 'AAPL', 'MSFT', 'GOOGL').
    * `START_DATE`, `END_DATE`: The historical period for your data.
    * `PREDICTION_DAYS`: How many days ahead the model should predict.
    * `TEST_SIZE`: The proportion of data used for testing and backtesting the ML model.
    * `INITIAL_CAPITAL`: Your starting capital for the simulation.
    * `TRADE_SIZE_PERCENT`: The percentage of capital allocated per trade.
    * `FEATURES`: Customize the list of technical indicators used as input for the ML model.

2.  **Run the backtester:**
    ```bash
    python main.py
    ```

    The script will fetch data, engineer features, train the ML model, run the backtest, print performance metrics, and display a plot of the portfolio's performance.

## Project Structure
