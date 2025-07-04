import pandas as pd
import joblib # saving the model, better for scikit-learn compatible models than pickle
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
    print(f"Features, ranked:")
    print(fi.sort_values('importance', ascending=False))

def train_model(train_df, feature_names, test_full, test_prediction):
    """Train XGBoost model using buffer data for validation"""
    xgb_params = {
        'base_score': 0.5, 
        'booster': 'gbtree',      
        'n_estimators': 1000,
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'learning_rate': 0.01,
        'early_stopping_rounds': 50
    }
    
    X_train = train_df[feature_names]
    y_train = train_df['Load']
    
    # Use all testing data for validation during training
    X_val = test_full[feature_names]
    y_val = test_full['Load']
    
    # Prepare prediction window features
    X_pred = test_prediction[feature_names]
    y_pred = test_prediction['Load']
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=100)
    
    return model, X_pred, y_pred

def getPrediction(model, test_prediction_df, X_pred):
    """Make predictions only on the prediction window"""
    test_prediction_df = test_prediction_df.copy()
    predictions = model.predict(X_pred)
    test_prediction_df['prediction'] = predictions
    
    return predictions

def forecast_on_prediction_window(df, test_prediction, model, X_pred):
    """Show forecast only on the prediction window within full dataset context"""
    test_prediction = test_prediction.copy()
    test_prediction['prediction'] = model.predict(X_pred)
    
    # Merge with full dataset to show context
    df_with_pred = df.copy()
    df_with_pred = df_with_pred.merge(test_prediction[['prediction']], 
                                     how='left', left_index=True, right_index=True)
    
    # Plot full dataset with predictions highlighted
    ax = df_with_pred[['Load']].plot(figsize=(15, 5))
    df_with_pred['prediction'].plot(ax=ax, style='.', markersize=8, color='red')
    plt.legend(['Historical Data', 'Predictions'])
    ax.set_title('Full Dataset with Prediction Window Highlighted')
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
    """Split data ensuring lag features don't leak into training"""
    df['Date'] = pd.to_datetime(df['Date'])
    # Take 2x the testlength in days so that the prediction window can reference 
    # the lag features in the past week, without referencing training data.
    train_end_date = df['Date'].max() - pd.Timedelta(days=2*testLength_days)
    
    train = df[df['Date'] < train_end_date].copy()
    test_full = df[(df['Date'] >= train_end_date)].copy()
    
    # Split the test set into buffer and actual prediction window
    prediction_start_date = df['Date'].max() - pd.Timedelta(days=testLength_days)
    test_buffer = test_full[test_full['Date'] < prediction_start_date].copy()
    test_prediction = test_full[test_full['Date'] >= prediction_start_date].copy()
    
    train = train.drop(['Date'], axis=1, errors='ignore')
    test_buffer = test_buffer.drop(['Date'], axis=1, errors='ignore')
    #test_prediction = test_prediction.drop(['Date'], axis=1, errors='ignore') // Using this to find problematic dates in sliding window
    
    print(f"Split sizes - Train: {len(train)}, Buffer: {len(test_buffer)}, Prediction: {len(test_prediction)}")
    
    return train, test_full, test_buffer, test_prediction

def plot_xgboost_forecast_vs_actual(test_df, predictions, smoothed_predictions):
    plt.figure(figsize=(15, 5))
    plt.plot(test_df.index, test_df['Load'], label='Actual Load')
    plt.plot(test_df.index, predictions, label='Forecast', color='red')
    plt.plot(test_df.index, smoothed_predictions, label='Smoothed Forecast', color='green')
    plt.title("XGBoost: Forecast vs Actual (Last Week)")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

def process_single_split(df, train, test_full, test_buffer, test_prediction):
    """Process split with separate buffer and prediction windows"""
    exclude_cols = ['Load', 'Date', 'Time Stamp', 'Load_1WeekAgo']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print("Training model...")
    model, X_pred, y_pred = train_model(train, feature_cols, test_full, test_prediction)
    
    return model, X_pred, y_pred

def expanding_window(df):
    splits = [7, 14, 30, 90, 365, 1095, 1825] # 1W, 2W, 1M, 3M, 1Y, 3Y, 5Y splits
    squareErrors = []
    for i in range(0, len(splits)):
        train, test = split(df, splits[i])
        model, X_test, y_test = process_single_split(df, train, test)
        predictions = getPrediction(model, test, X_test)
        rmse, mae, mape = evaluate_forecast(predictions, y_test)
        squareErrors.append(rmse)
    
    averageRMSE = sum(squareErrors) / len(squareErrors)
    return averageRMSE

def sliding_window(df):
    RMSE_array = []
    dates_over_threshold = []
    min_required_rows = 24 * (7 + 7*2)  # example, depends on your train/test split size

    for i in range(52):
        if i > 0:
            df_window = df[:-720*i]
        else:
            df_window = df.copy()
        
        if len(df_window) < min_required_rows:
            print(f"Iteration {i} skipped: not enough data ({len(df_window)} rows).")
            continue
        
        model, train, test_full, test_buffer, test_prediction, X_pred, y_pred = getNDayModel(df_window, 7)
        predictions = getPrediction(model, test_prediction, X_pred)
        
        if len(predictions) == 0 or len(y_pred) == 0:
            print(f"Iteration {i} skipped: no predictions or actuals.")
            continue

        rmse, mae, mape = evaluate_forecast(predictions, y_pred)

        if rmse > 250:
            test_prediction = test_prediction.copy()
            test_prediction['prediction'] = predictions
            test_prediction['error'] = np.abs(test_prediction['Load'] - test_prediction['prediction'])
            test_prediction['Date'] = pd.to_datetime(test_prediction['Date'])

            for date, group in test_prediction.groupby(test_prediction['Date'].dt.date):
                if (group['error'] > 300).any():
                    dates_over_threshold.append(date)

        RMSE_array.append(rmse)

    print(f"Mean RMSE: {sum(RMSE_array)/len(RMSE_array) if RMSE_array else float('nan')}")
    print("Dates with large daily errors:", dates_over_threshold)
    return dates_over_threshold

def getNDayModel(df, NDays):
    """Get model that predicts only on the final N days"""
    train, test_full, test_buffer, test_prediction = split(df, NDays)
    model, X_pred, y_pred = process_single_split(df, train, test_full, test_buffer, test_prediction)
    
    return model, train, test_full, test_buffer, test_prediction, X_pred, y_pred

def main():
    path = '../data/processed/df.csv'
    df = pd.read_csv(path)
    print(df.dtypes)
    #Sliding window of 1 week forecasts:
    sliding_window(df)
"""
    #Plotting the last week of the data against a forecast (use case)
    predictionLength = 7
    print(f"Getting final model trained on all but {predictionLength*2} days, predicting on final {predictionLength} days...")
    
    model, train, test_full, test_buffer, test_prediction, X_pred, y_pred = getNDayModel(df, predictionLength)

    print(f"Saving {predictionLength} day model...")
    joblib.dump(model, 'xgboost_model.pkl')

    print("Predicting on prediction window only...")
    predictions = getPrediction(model, test_prediction, X_pred)

    print("Evaluating forecast on prediction window...")
    rmse, mae, mape = evaluate_forecast(predictions, y_pred)

    print("Displaying forecast in context of full dataset...")
    forecast_on_prediction_window(df, test_prediction, model, X_pred)
    
    feature_importance(model)

    print(f"Plotting {predictionLength} day forecast...")
    smoothed_predictions = pd.Series(predictions).rolling(window=3, center=True).mean().bfill().ffill().to_numpy()
    plot_xgboost_forecast_vs_actual(test_prediction, predictions, smoothed_predictions)
"""

if __name__ == "__main__":
    main()

"""
xgb_params = {
        'base_score': 0.5, 
        'booster': 'gbtree',      
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'learning_rate': 0.01,
    }
"""

