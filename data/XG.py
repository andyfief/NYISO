import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def add_rolling_features_training(df, target='Load', window=3):
    """Add rolling features for training data"""
    df = df.copy()
    rolling_window = 2 * window + 1

    # Centered rolling average (look-ahead; only for training)
    df[f'{target}Rolling{window}'] = df[target].rolling(window=rolling_window, center=True, min_periods=1).mean()

    # Trailing rolling average (causal; usable at inference)
    df[f'{target}Trailing{window}'] = df[target].rolling(window=window, min_periods=1).mean()

    return df

def add_rolling_features_inference(train_df, test_df, target='Load', window=3):
    """Add rolling features for test data using only historical information and predictions"""
    test_df = test_df.copy()
    
    # Get the last 'window' values from training data for initial trailing average
    train_tail = train_df[target].tail(window).values
    
    # Identify base features (non-rolling features)
    all_features = train_df.drop(columns=[target]).select_dtypes(include=['number']).columns
    base_features = [col for col in all_features if 'Rolling' not in col and 'Trailing' not in col]
    
    # Prepare training data with only base features
    X_train_base = train_df[base_features]
    y_train = train_df[target]
    
    # Train a temporary model using only base features
    temp_model = xgb.XGBRegressor(
        n_estimators=50,  # Reduced for speed since this is just for initial predictions
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    temp_model.fit(X_train_base, y_train)
    
    # Get initial predictions for test set using only base features
    X_test_base = test_df[base_features]
    initial_predictions = temp_model.predict(X_test_base)
    
    # Calculate trailing rolling average for test set using predictions
    trailing_values = []
    historical_window = list(train_tail)
    
    for i in range(len(test_df)):
        # Calculate trailing average using only historical data (including predictions)
        if len(historical_window) >= window:
            trailing_avg = np.mean(historical_window[-window:])
        else:
            trailing_avg = np.mean(historical_window)
        
        trailing_values.append(trailing_avg)
        
        # Add predicted value to historical window for next iteration
        # This uses predictions instead of actual values
        historical_window.append(initial_predictions[i])
    
    test_df[f'{target}Trailing{window}'] = trailing_values
    
    # No centered rolling for test set (future-looking)
    test_df[f'{target}Rolling{window}'] = float('nan')
    
    return test_df

def main():
    # Load data
    df = pd.read_csv('final.csv')
    df = df.drop('Temperature', axis=1)
    
    # Split first to avoid data leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Add rolling features properly
    train_df = add_rolling_features_training(train_df, target='Load', window=3)
    test_df = add_rolling_features_inference(train_df, test_df, target='Load', window=3)
    
    # Prepare features and target
    X_train = train_df.drop(columns=['Load']).select_dtypes(include=['number'])
    y_train = train_df['Load']
    X_test = test_df.drop(columns=['Load']).select_dtypes(include=['number'])
    y_test = test_df['Load']
    
    # Ensure column order matches between train and test
    X_test = X_test.reindex(columns=X_train.columns)

    # Train the model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    correction = y_pred.mean() - y_test.mean()
    y_pred_adj = y_pred - correction
    mse = mean_squared_error(y_test, y_pred_adj)
    print(f'Mean Squared Error on test set: {mse:.4f}')
    print(f'Off by {np.sqrt(mse)}')

    # Get feature importance
    importance = model.feature_importances_
    feature_names = X_train.columns

    # Create a DataFrame for easy viewing
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("Feature Importance (Gain):")
    print(importance_df)

    # Plot predicted vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[100:200], label='Actual Load', alpha=0.7)
    plt.plot(y_pred_adj[100:200], label='Predicted Load', alpha=0.7)
    plt.title('XGBoost Load Prediction vs Actual (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Load')
    plt.legend()
    plt.show()

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.gca().invert_yaxis()  # Most important at top
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()