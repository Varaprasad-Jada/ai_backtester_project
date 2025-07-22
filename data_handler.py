# data_handler.py
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' data,
                      or None if data fetching fails.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"Warning: No data found for {ticker} from {start_date} to {end_date}.")
            return None
        print(f"Successfully fetched {len(data)} rows for {ticker}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def prepare_data(df):
    """
    Prepares the fetched data by ensuring 'Close' column exists and handling NaNs.

    Args:
        df (pd.DataFrame): Raw DataFrame from yfinance.

    Returns:
        pd.DataFrame: Prepared DataFrame.
    """
    if 'Close' not in df.columns:
        print("Error: 'Close' column not found in data.")
        return None
    df = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    df.fillna(method='ffill', inplace=True) # Forward-fill NaNs
    df.fillna(method='bfill', inplace=True) # Backward-fill any remaining NaNs
    if df.isnull().sum().sum() > 0:
        print("Warning: NaNs still present after filling. Dropping rows with NaNs.")
        df.dropna(inplace=True)
    return df

if __name__ == '__main__':
    # Example Usage:
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2023-12-31'
    
    raw_data = fetch_data(ticker, start, end)
    if raw_data is not None:
        processed_data = prepare_data(raw_data)
        if processed_data is not None:
            print("\nSample Processed Data Head:")
            print(processed_data.head())
            print("\nSample Processed Data Tail:")
            print(processed_data.tail())