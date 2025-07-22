# analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(results_df, initial_capital):
    """
    Calculates key performance metrics for the backtest.

    Args:
        results_df (pd.DataFrame): DataFrame from run_backtest.
        initial_capital (float): The starting capital.

    Returns:
        dict: A dictionary of performance metrics.
    """
    if results_df is None or results_df.empty:
        return {}

    total_return = (results_df['Portfolio_Value'].iloc[-1] - initial_capital) / initial_capital
    
    # Max Drawdown
    peak = results_df['Portfolio_Value'].expanding(min_periods=1).max()
    drawdown = (results_df['Portfolio_Value'] - peak) / peak
    max_drawdown = drawdown.min()

    # Annualized Return (assuming daily data for simplicity, adjust for other frequencies)
    # Factor based on total days in backtest vs trading days in a year (approx 252)
    num_trading_days = len(results_df)
    if num_trading_days > 0:
        annualized_return = (1 + total_return) ** (252 / num_trading_days) - 1
    else:
        annualized_return = 0

    # Sharpe Ratio (Requires risk-free rate, simplified here)
    # For a real Sharpe, you'd need daily returns and a risk-free rate
    returns = results_df['Portfolio_Value'].pct_change().dropna()
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) # Assuming daily returns for sqrt(252)
    else:
        sharpe_ratio = 0 # No volatility, or all returns are same

    metrics = {
        'Initial Capital': f"${initial_capital:,.2f}",
        'Final Portfolio Value': f"${results_df['Portfolio_Value'].iloc[-1]:,.2f}",
        'Total Return (%)': f"{total_return * 100:.2f}%",
        'Annualized Return (%)': f"{annualized_return * 100:.2f}%",
        'Max Drawdown (%)': f"{max_drawdown * 100:.2f}%",
        'Sharpe Ratio (Annualized, Simplified)': f"{sharpe_ratio:.2f}"
    }
    return metrics

def plot_results(results_df, ticker_data_for_plot=None):
    """
    Plots the portfolio value and optionally the stock price with signals.

    Args:
        results_df (pd.DataFrame): DataFrame with 'Portfolio_Value' from backtest.
        ticker_data_for_plot (pd.DataFrame, optional): Original DataFrame with 'Close'
                                                      and 'Predicted_Signal' to plot signals.
    """
    if results_df is None or results_df.empty:
        print("No results to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Portfolio Value
    ax1.plot(results_df.index, results_df['Portfolio_Value'], label='Portfolio Value', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('AI-Driven Backtest Performance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')

    if ticker_data_for_plot is not None and 'Close' in ticker_data_for_plot.columns and 'Predicted_Signal' in ticker_data_for_plot.columns:
        # Filter ticker_data_for_plot to align with the backtested period for signals
        plot_df = ticker_data_for_plot.loc[results_df.index.min():results_df.index.max()]
        
        ax2 = ax1.twinx() # Create a second y-axis
        ax2.plot(plot_df.index, plot_df['Close'], label='Stock Close Price', color='gray', linestyle='--', alpha=0.6)
        ax2.set_ylabel('Stock Price ($)', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Plot buy/sell signals on the stock price chart
        buy_signals = plot_df[plot_df['Predicted_Signal'] == 1]
        ax2.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal', alpha=0.8)
        
        # If you implemented sell signals (e.g., Predicted_Signal == 0 for sell)
        # sell_signals = plot_df[plot_df['Predicted_Signal'] == 0]
        # ax2.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal', alpha=0.8)

        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Example Usage (requires a DataFrame with 'Portfolio_Value' and optionally 'Close', 'Predicted_Signal')
    # Create dummy data for testing analyzer.py in isolation
    dummy_backtest_results = pd.DataFrame({
        'Portfolio_Value': np.linspace(100000, 110000, 100) + np.sin(np.linspace(0, 10, 100)) * 5000
    })
    dummy_backtest_results.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D'))
    
    dummy_ticker_data = pd.DataFrame({
        'Close': np.linspace(150, 160, 100) + np.cos(np.linspace(0, 10, 100)) * 5,
        'Predicted_Signal': np.random.randint(0, 2, 100)
    })
    dummy_ticker_data.index = dummy_backtest_results.index

    metrics = calculate_metrics(dummy_backtest_results, 100000)
    print("\nCalculated Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    plot_results(dummy_backtest_results, dummy_ticker_data)