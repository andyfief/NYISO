import pandas as pd 
import xgboost as xgb  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
from datetime import timedelta

def main(test_date='2025-03-20'):
    """
    Main function for weekly load forecasting using XGBoost.
    
    Args:
        test_date (str): Optional. A specific date to simulate as "current date" to test the model.
                         If None, it uses a basic 80/20 time-based split.
    """
    
    # Load pre-processed dataset (feature engineering already done) from CSV
    df = pd.read_csv('final.csv')
    
    # If the dataset includes a 'Date' column, convert it to datetime format
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print('No date column in dataframe')
        return
    
    # Create the target variable: load value 7 days ahead
    # `shift(-7)` moves the Load value 7 days earlier (to match with today's features)
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
    df['Timestamp_plus_7d'] = df['Time Stamp'] + pd.Timedelta(days=7)

    df_target = df[['Time Stamp', 'Load']].copy()
    df_target.rename(columns={'Time Stamp': 'Timestamp_plus_7d', 'Load': 'Load_target'}, inplace=True)

    df = df.merge(df_target, on='Timestamp_plus_7d', how='left')
    
    # Drop the rows at the end that now have NaN in the new target column
    df = df.dropna(subset=['Load_target'])
    
    test_date = pd.to_datetime(test_date)
    
    
    test_date = pd.to_datetime(test_date)
    cutoff_date = test_date - timedelta(days=7)


    # Train on everything before the cutoff
    train_df = df[df['Date'] < cutoff_date]

    # Test only on the last 7 days (cutoff_date to test_date inclusive)
    test_df = df[(df['Date'] >= cutoff_date) & (df['Date'] < test_date)]
    drop_cols = ['LoadTrailing1', 'LoadTrailing3', 'LoadTrailing7', 'LoadDayRolling3', 'LoadDayRolling7', 'LoadDayRolling14', '1DayLag']
    
    test_df = test_df.drop(columns=drop_cols, errors='ignore')

    test_range_days = test_df['Date'].dt.normalize().nunique() # debugging purposes only.
    assert test_range_days == 7, f"Expected 7 unique test days, got {test_range_days}"

    print(f"=== Simulating forecast as of {test_date.strftime('%Y-%m-%d')} ===")
    print(f"Training data: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Test data: {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Define which columns to exclude from the model input
    exclude_cols = ['Load', 'Load_target', 'Date']  # Exclude actual load and target and date
    feature_cols = [col for col in df.columns if col not in exclude_cols + drop_cols]
    
    # Keep only numeric columns (non-numeric will break XGBoost)
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Prepare input (X) and output (y) for training and testing
    X_train = train_df[numeric_features]
    y_train = train_df['Load_target']
    X_test = test_df[numeric_features]
    y_test = test_df['Load_target']
    
    # Save the list of features for later use (for plotting/importances)
    feature_cols = numeric_features
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    
    # Initialize the XGBoost regressor model
    model = xgb.XGBRegressor(
        n_estimators=200,          # number of trees
        max_depth=6,               # how deep each tree can go (controls model complexity)
        learning_rate=0.1,         # how much each tree adjusts the model (lower = slower training)
        objective='reg:squarederror',  # regression problem with squared error loss
        random_state=42,           # ensures reproducibility
        early_stopping_rounds=20   # stop training if no improvement after 20 rounds
    )
    
    # Define validation set (used for early stopping)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=eval_set,  # used for evaluating progress
        verbose=False        # set to True if you want to see training logs
    )
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error (same units as target)
    mae = mean_absolute_error(y_test, y_pred)  # Average absolute error
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
    
    # Display the results
    print(f"\n=== 7-Day Ahead Forecast Results ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Get and display feature importances
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_  # how important each feature was in the trees
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== Top 10 Most Important Features ===")
    print(importance_df.head(17))
    
    # Plot actual vs predicted for first 200 points
    plt.figure(figsize=(10, 6))
    sample_size = min(200, len(y_test))
    plt.plot(y_test.values[:sample_size], label='Actual Load', alpha=0.7)
    plt.plot(y_pred[:sample_size], label='Predicted Load', alpha=0.7)
    plt.title('7-Day Ahead Forecast vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Load')
    plt.legend()

    daily_actual = test_df.copy()
    daily_actual['Predicted'] = y_pred
    daily_actual['Actual'] = y_test.values
    daily_actual['DateOnly'] = daily_actual['Date'].dt.normalize()

    # Compute daily mean values
    daily_means = (
        daily_actual.groupby('DateOnly')[['Actual', 'Predicted']]
        .mean()
        .reset_index()
    )
    
    # Plot daily rolling comparison
    plt.figure(figsize=(10, 5))
    plt.plot(daily_means['DateOnly'], daily_means['Actual'], label='Actual Daily Mean', marker='o')
    plt.plot(daily_means['DateOnly'], daily_means['Predicted'], label='Predicted Daily Mean', marker='x')
    plt.title('Daily Average Load: Actual vs Predicted (7-Day Test Period)')
    plt.xlabel('Date')
    plt.ylabel('Average Load')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Optional: Feature importance bar plot (disabled in this version)
    """
    plt.subplot(2, 2, 3)
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances')
    plt.gca().invert_yaxis()
    """
    
    plt.show()
    
    # Save the model to a file (optional)
    # model.save_model('weekly_forecast_model.json')
    
    return model, importance_df


# Run the main function if this script is executed directly
if __name__ == "__main__":
    # Train and evaluate the model using 80/20 split
    model, feature_importance = main()
    
    # Optional: Run with specific test date (simulate forecasting on that day)
    # model_2023, fi_2023 = main(test_date='2023-06-01')
    
    # Optional: Evaluate on multiple dates to test robustness
    # test_dates = ['2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01']
    # for date in test_dates:
    #     print(f"\n{'='*50}")
    #     model_test, _ = main(test_date=date)
