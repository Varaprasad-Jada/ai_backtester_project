# ml_model.py
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

def train_predict_model(df, features_list, target_column='Target', test_size=0.2, random_state=42):
    """
    Trains a RandomForestClassifier and makes predictions.

    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        features_list (list): List of column names to use as features (X).
        target_column (str): Name of the target column (y).
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (pd.DataFrame with 'Predicted_Signal' column, trained model, X_test, y_test)
               Returns (None, None, None, None) if inputs are invalid.
    """
    if not all(col in df.columns for col in features_list + [target_column]):
        print("Error: One or more specified features or target column not found in DataFrame.")
        return None, None, None, None

    X = df[features_list]
    y = df[target_column]

    # Split data into training and testing sets while preserving order for time series data
    # We typically use a time-based split for financial data to avoid look-ahead bias.
    # Here, a simple sequential split is used for demonstration.
    # For robust backtesting, often a rolling window or fixed-period training is done.
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Data split: Train size = {len(X_train)}, Test size = {len(X_test)}")

    # Initialize and train the RandomForestClassifier
    # Hyperparameters can be tuned for better performance
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced') # Use balanced for imbalanced targets
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = model.predict(X_test)
    
    # Add predictions back to the original DataFrame for easier merging with backtester
    df_with_predictions = df.copy()
    df_with_predictions['Predicted_Signal'] = np.nan # Initialize with NaN
    df_with_predictions.loc[X_test.index, 'Predicted_Signal'] = predictions

    print("\nModel Training Complete.")
    print(f"Accuracy on test set: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report on test set:")
    print(classification_report(y_test, predictions))

    return df_with_predictions, model, X_test, y_test

if __name__ == '__main__':
    # Example Usage (requires a DataFrame with features and target)
    # You'd typically load and feature engineer data first
    sample_features = ['SMA_10', 'RSI', 'MACD'] # Example features
    
    # Create a dummy DataFrame for testing ml_model.py in isolation
    dummy_data = pd.DataFrame({
        'SMA_10': np.random.rand(100) * 10,
        'RSI': np.random.rand(100) * 100,
        'MACD': np.random.rand(100) * 5,
        'Target': np.random.randint(0, 2, 100) # Random 0s and 1s
    })
    dummy_data.index = pd.to_datetime(pd.date_range(start='2022-01-01', periods=100, freq='D'))
    
    df_pred, model, X_test, y_test = train_predict_model(dummy_data, sample_features)
    if df_pred is not None:
        print("\nDataFrame with Predicted Signals Head:")
        print(df_pred[['SMA_10', 'RSI', 'MACD', 'Target', 'Predicted_Signal']].head())
        print("\nDataFrame with Predicted Signals Tail (showing actual predictions):")
        print(df_pred[['SMA_10', 'RSI', 'MACD', 'Target', 'Predicted_Signal']].tail())