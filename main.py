# main.py
from data_handler import fetch_data, prepare_data
from feature_engineer import add_technical_indicators, create_target_variable
from ml_model import train_predict_model
from backtester import run_backtest
from analyzer import calculate_metrics, plot_results

import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """
    Orchestrates the AI-driven backtesting process.
    """
    print("--- Starting AI-Driven Backtester ---")

    # --- Configuration ---
    STOCK_TICKER = 'MSFT' # Example: Microsoft
    START_DATE = '2018-01-01'
    END_DATE = '2023-12-31'
    PREDICTION_DAYS = 1 # Predict 1 day ahead
    TEST_SIZE = 0.20 # 20% of data for testing/backtesting
    INITIAL_CAPITAL = 100000
    TRADE_SIZE_PERCENT = 0.10 # Allocate 10% of capital per trade

    # Define features to be used by the ML model
    FEATURES = [
        'SMA_10', 'SMA_20', 'SMA_50',
        'RSI', 'MACD', 'Signal_Line',
        'BBLower', 'BBUpper', 'ROC', 'Volume_ROC'
    ]
    TARGET_COLUMN = 'Target'

    # --- 1. Data Acquisition ---
    print(f"\n1. Fetching data for {STOCK_TICKER} from {START_DATE} to {END_DATE}...")
    raw_df = fetch_data(STOCK_TICKER, START_DATE, END_DATE)
    if raw_df is None or raw_df.empty:
        print("Exiting: Failed to fetch or prepare data.")
        return

    processed_df = prepare_data(raw_df)
    if processed_df is None or processed_df.empty:
        print("Exiting: Data preparation failed.")
        return

    # --- 2. Feature Engineering & Target Variable Creation ---
    print("\n2. Adding technical indicators and creating target variable...")
    df_with_features = add_technical_indicators(processed_df)
    if df_with_features is None or df_with_features.empty:
        print("Exiting: Feature engineering failed.")
        return

    df_final = create_target_variable(df_with_features, forward_days=PREDICTION_DAYS)
    if df_final is None or df_final.empty:
        print("Exiting: Target variable creation failed.")
        return

    # Ensure features exist after dropping NaNs for target creation
    final_features = [f for f in FEATURES if f in df_final.columns]
    if not final_features:
        print("Error: No valid features remaining after data preparation. Check feature list and data.")
        return

    # --- 3. Machine Learning Model Training & Prediction ---
    print("\n3. Training ML model and making predictions...")
    df_with_predictions, model, X_test, y_test = train_predict_model(
        df_final, final_features, target_column=TARGET_COLUMN, test_size=TEST_SIZE
    )

    if df_with_predictions is None:
        print("Exiting: ML model training or prediction failed.")
        return
    
    # Filter the DataFrame to only include the backtesting period (where predictions exist)
    backtest_data = df_with_predictions.dropna(subset=['Predicted_Signal'])
    if backtest_data.empty:
        print("Exiting: No data points with predictions for backtesting. Adjust test_size or data range.")
        return

    # --- 4. Backtesting the Strategy ---
    print("\n4. Running backtest with AI signals...")
    results_df = run_backtest(backtest_data, initial_capital=INITIAL_CAPITAL, trade_size_percent=TRADE_SIZE_PERCENT)
    
    if results_df is None or results_df.empty:
        print("Exiting: Backtest failed or produced no results.")
        return

    # --- 5. Analyze and Visualize Results ---
    print("\n5. Analyzing and visualizing results...")
    metrics = calculate_metrics(results_df, INITIAL_CAPITAL)
    
    print("\n--- Backtest Performance Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n--- Plotting Results ---")
    plot_results(results_df, df_with_predictions) # Pass df_with_predictions to plot actual signals

    print("\n--- Backtester Finished ---")

if __name__ == '__main__':
    main()