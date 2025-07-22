# feature_engineer.py
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds common technical indicators to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'Close', 'High', 'Low', 'Volume' columns.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    df = df.copy() # Work on a copy
    
    # Simple Moving Averages (SMA)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BBLower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
    df['BBUpper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
    
    # Price Rate of Change (ROC)
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

    # Volume Rate of Change
    df['Volume_ROC'] = ((df['Volume'] - df['Volume'].shift(10)) / df['Volume'].shift(10)) * 100

    # Drop initial NaN rows created by rolling windows
    df.dropna(inplace=True)
    
    print(f"Added technical indicators. DataFrame shape: {df.shape}")
    return df

def create_target_variable(df, forward_days=1):
    """
    Creates the target variable: 1 if price goes up, 0 if it goes down/stays same.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        forward_days (int): Number of days forward to predict.

    Returns:
        pd.DataFrame: DataFrame with a new 'Target' column.
    """
    df = df.copy() # Work on a copy
    # Shift 'Close' price by `forward_days` to get the future price
    df['Future_Close'] = df['Close'].shift(-forward_days)
    # Target is 1 if future close is higher, 0 otherwise
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    
    # Drop rows where Future_Close (and thus Target) is NaN (at the end of the dataset)
    df.dropna(subset=['Future_Close', 'Target'], inplace=True)
    
    print(f"Created target variable for {forward_days} day(s) ahead. DataFrame shape: {df.shape}")
    return df

if __name__ == '__main__':
    # Example Usage (requires a DataFrame with 'Close', 'High', 'Low', 'Volume')
    # You'd typically load data using data_handler.py first
    sample_data = pd.DataFrame({
        'Close': np.random.rand(100) * 100 + 50,
        'High': np.random.rand(100) * 100 + 55,
        'Low': np.random.rand(100) * 100 + 45,
        'Volume': np.random.rand(100) * 1000000
    })
    sample_data.index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
    
    df_with_features = add_technical_indicators(sample_data)
    if df_with_features is not None:
        df_final = create_target_variable(df_with_features)
        if df_final is not None:
            print("\nSample DataFrame with Features and Target Head:")
            print(df_final.head())
            print("\nSample DataFrame with Features and Target Tail:")
            print(df_final.tail())
            print("\nTarget variable distribution:")
            print(df_final['Target'].value_counts())