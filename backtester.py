# backtester.py
import pandas as pd
import numpy as np

def run_backtest(df_with_signals, initial_capital=100000, trade_size_percent=0.10):
    """
    Runs a simple backtest based on AI predictions.

    Assumptions:
    - Trades are executed at the 'Close' price of the signal day.
    - No slippage or transaction costs are modeled.
    - Only long positions (buy and sell to close/profit).
    - Trades are sized based on a percentage of current capital.

    Args:
        df_with_signals (pd.DataFrame): DataFrame containing 'Close' prices
                                         and 'Predicted_Signal' (1 for buy, 0 for hold/sell).
        initial_capital (float): Starting capital for the backtest.
        trade_size_percent (float): Percentage of capital to allocate per trade.

    Returns:
        pd.DataFrame: DataFrame with backtest results ('Cash', 'Holdings', 'Portfolio_Value').
    """
    if 'Predicted_Signal' not in df_with_signals.columns or 'Close' not in df_with_signals.columns:
        print("Error: 'Predicted_Signal' or 'Close' column missing for backtesting.")
        return None

    # Ensure signals are only available from the test set period, NaNs before that.
    # Drop rows where 'Predicted_Signal' is NaN, as we can't trade without a signal.
    # This also effectively starts the backtest from where the test set begins.
    df = df_with_signals.dropna(subset=['Predicted_Signal']).copy()

    capital = initial_capital
    holdings = 0
    portfolio_value = initial_capital
    
    cash_history = []
    holdings_history = []
    portfolio_value_history = []
    
    print("\nStarting Backtest...")
    for index, row in df.iterrows():
        current_price = row['Close']
        signal = row['Predicted_Signal']
        
        # --- Trading Logic ---
        if signal == 1: # Predicted Up: BUY signal
            if capital > 0: # Only buy if we have cash
                trade_amount_cash = capital * trade_size_percent
                num_shares_to_buy = trade_amount_cash / current_price
                
                holdings += num_shares_to_buy
                capital -= trade_amount_cash # Deduct cash spent
                
                # print(f"Date: {index.strftime('%Y-%m-%d')} - BUY {num_shares_to_buy:.2f} shares at {current_price:.2f}")
        
        # You could add a sell/short signal (e.g., if signal == 0 or -1 for 'down')
        # For simplicity, this example only covers buying and holding or selling implicitly.
        # If signal implies 'sell existing position', then you'd add:
        # elif signal == 0 and holdings > 0:
        #     capital += holdings * current_price
        #     holdings = 0
        #     print(f"Date: {index.strftime('%Y-%m-%d')} - SELL ALL shares at {current_price:.2f}")

        # --- Update Portfolio Value ---
        portfolio_value = capital + (holdings * current_price)
        
        cash_history.append(capital)
        holdings_history.append(holdings)
        portfolio_value_history.append(portfolio_value)

    results_df = pd.DataFrame({
        'Cash': cash_history,
        'Holdings': holdings_history,
        'Portfolio_Value': portfolio_value_history
    }, index=df.index) # Use the index of the backtested data

    print("Backtest Complete.")
    return results_df

if __name__ == '__main__':
    # Example Usage (requires a DataFrame with 'Close' and 'Predicted_Signal')
    # Create a dummy DataFrame for testing backtester.py in isolation
    dummy_data_for_backtest = pd.DataFrame({
        'Close': np.random.rand(50) * 100 + 50,
        'Predicted_Signal': np.random.randint(0, 2, 50) # Random 0s and 1s
    })
    dummy_data_for_backtest.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=50, freq='D'))
    
    backtest_results = run_backtest(dummy_data_for_backtest)
    if backtest_results is not None:
        print("\nSample Backtest Results Head:")
        print(backtest_results.head())
        print("\nSample Backtest Results Tail:")
        print(backtest_results.tail())