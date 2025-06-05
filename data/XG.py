import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

def main(test_date='2025-03-30'):
    """
    Main function for weekly load forecasting
    
    Args:
        test_date (str): Date to simulate as "current date" for testing (format: 'YYYY-MM-DD')
                        If None, uses the standard 80/20 split
    """
    # Load data (assumes all feature engineering is done in CSV)
    df = pd.read_csv('final.csv')
    
    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Create target: Load value 7 days ahead
    df['Load_target'] = df['Load'].shift(-7)
    
    # Remove rows where target is NaN (last week of data)
    df = df.dropna(subset=['Load_target'])
    
    # Split data based on test_date or default split
    if test_date is not None:
        test_date = pd.to_datetime(test_date)
        
        # Only use data up to test_date for training (simulate "current date")
        available_data = df[df['Date'] <= test_date]
        
        # Use last 20% of available data for testing
        split_idx = int(len(available_data) * 0.8)
        train_df = available_data.iloc[:split_idx]
        test_df = available_data.iloc[split_idx:]
        
        print(f"=== Simulating forecast as of {test_date.strftime('%Y-%m-%d')} ===")
        print(f"Training data: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"Test data: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
        
    else:
        # Standard chronological split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print("=== Using standard 80/20 split ===")
        if 'Date' in df.columns:
            print(f"Training data: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
            print(f"Test data: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Prepare features and target (exclude non-numeric columns)
    exclude_cols = ['Load', 'Load_target', 'Date']  # Add other non-numeric columns as needed
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Ensure all feature columns are numeric
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X_train = train_df[numeric_features]
    y_train = train_df['Load_target']
    X_test = test_df[numeric_features]
    y_test = test_df['Load_target']
    
    feature_cols = numeric_features  # Update for later use
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Train the model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42,
        early_stopping_rounds=20
    )
    
    # Fit with validation set for early stopping
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n=== 7-Day Ahead Forecast Results ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== Top 10 Most Important Features ===")
    print(importance_df.head(10))
    
    # Plotting
    plt.figure(figsize=(10, 6))

    sample_size = min(200, len(y_test))
    plt.plot(y_test.values[:sample_size], label='Actual Load', alpha=0.7)
    plt.plot(y_pred[:sample_size], label='Predicted Load', alpha=0.7)
    plt.title('7-Day Ahead Forecast vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Load')
    plt.legend()
    
    """
    # Plot 3: Feature importance (top 15)
    plt.subplot(2, 2, 3)
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    """

    plt.show()
    
    # Save model (optional)
    # model.save_model('weekly_forecast_model.json')
    
    return model, importance_df

if __name__ == "__main__":
    # Example usage:
    
    # Standard run (80/20 split on all data)
    model, feature_importance = main()
    
    # Test on a specific date (simulate forecasting as of that date)
    # model_2023, fi_2023 = main(test_date='2023-06-01')
    
    # Test multiple dates for robustness
    # test_dates = ['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01']
    # for date in test_dates:
    #     print(f"\n{'='*50}")
    #     model_test, _ = main(test_date=date)