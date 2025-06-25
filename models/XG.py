import pandas as pd 
import xgboost as xgb  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime, timedelta

def feature_importance(model):
    print("Calculating feature importance...")
    fi = pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
             columns=['importance'])
    print(f"Top 5 most important features:")
    print(fi.sort_values('importance', ascending=False).head())
    fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
    plt.show()
    print("Feature importance plot displayed.")

def train_model(train_df, feature_names, test_df):
    """Train XGBoost model on training data"""
    xgb_params = {
        'base_score': 0.5, 
        'booster': 'gbtree',      
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'objective': 'reg:linear',
        'max_depth': 3,
        'learning_rate': 0.01
    }
    
    X_train = train_df[feature_names]
    y_train = train_df['Load']
    X_test = test_df[feature_names]
    y_test = test_df['Load']  # Fixed: changed from 'load' to 'Load'
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)
    
    return model, X_test, y_test  # Added y_test to return

def predict(model, test_df, X_test):
    predictions = model.predict(X_test)
    test_df['prediction'] = predictions
    
    # Fixed plotting logic
    ax = test_df[['Load']].plot(figsize=(15, 5))
    test_df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Data and Prediction')
    plt.show()
    
    return predictions

def forecast_on_test(df, test, model, X_test):
    test['prediction'] = model.predict(X_test)
    df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
    ax = df[['Load']].plot(figsize=(15, 5))
    df['prediction'].plot(ax=ax, style='.')
    plt.legend(['Truth Data', 'Predictions'])
    ax.set_title('Raw Dat and Prediction')
    plt.show()

def evaluate_forecast(predictions, actuals):
    """Evaluate overall forecast performance"""
    # Convert to numpy arrays for consistent handling
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"\nOverall Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Calculate daily RMSE progression (assuming 24 hours per day)
    hours_per_day = 24
    num_days = len(predictions) // hours_per_day
    
    if num_days > 0:
        print(f"\nDaily RMSE progression:")
        daily_errors = []
        for day in range(num_days):
            start_idx = day * hours_per_day
            end_idx = (day + 1) * hours_per_day
            
            daily_predictions = predictions[start_idx:end_idx]
            daily_actuals = actuals[start_idx:end_idx]
            
            daily_rmse = np.sqrt(mean_squared_error(daily_actuals, daily_predictions))
            daily_errors.append(daily_rmse)
            print(f"Day {day + 1}: {daily_rmse:.2f}")
    else:
        daily_errors = []
        print("Not enough data for daily breakdown.")
    
    return rmse, mae, mape


def split(df, testLength_days):
     # Create train/test split
    df['Date'] = pd.to_datetime(df['Date'])
    train_end_date = df['Date'].max() - pd.Timedelta(days=testLength_days)

    train = df[df['Date'] < train_end_date].copy()
    test = df[(df['Date'] >= train_end_date)].copy()
    
    train = train.drop(['Date'], axis=1, errors='ignore')
    print(f"Split sizes - Train: {len(train)}, Test: {len(test)}")

    return train, test

def main():
    path = '../data/processed/df.csv'
    df = pd.read_csv(path)

    testLength_days = 7
    train, test = split(df, testLength_days)

    exclude_cols = ['Load', 'Date', 'Time Stamp']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print("Training model...")
    model, X_test, y_test = train_model(train, feature_cols, test)

    print("Predicting...")
    predictions = predict(model, test, X_test)
    forecast_on_test(df, test, model, X_test)
    
    feature_importance(model)
    
    print("\nEvaluating forecast...")
    evaluate_forecast(predictions, y_test)

if __name__ == "__main__":
    main()
