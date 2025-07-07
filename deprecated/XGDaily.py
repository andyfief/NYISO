import pandas as pd
import joblib # saving the model, better for scikit-learn compatible models than pickle
import xgboost as xgb  
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 
import numpy as np
from datetime import datetime, timedelta

def feature_importance(model, day_num):
    print(f"Calculating feature importance for Day {day_num} model...")
    fi = pd.DataFrame(data=model.feature_importances_,
             index=model.feature_names_in_,
             columns=['importance'])
    print(f"Day {day_num} Features, ranked:")
    print(fi.sort_values('importance', ascending=False))
    return fi

def train_day_model(train_df, feature_names, validation_df, day_ahead):
    """Train XGBoost model for specific day ahead prediction"""
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
    
    # Use validation data for early stopping
    X_val = validation_df[feature_names]
    y_val = validation_df['Load']
    
    print(f"Training model for Day {day_ahead} prediction...")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)],
              verbose=False)  # Set to False to reduce output clutter
    
    return model

def get_day_prediction_window(test_prediction, day_ahead):
    """Get the specific 24-hour window for day N prediction"""
    hours_per_day = 24
    start_idx = (day_ahead - 1) * hours_per_day
    end_idx = day_ahead * hours_per_day
    
    if end_idx > len(test_prediction):
        end_idx = len(test_prediction)
        start_idx = max(0, end_idx - hours_per_day)
    
    return test_prediction.iloc[start_idx:end_idx].copy()

def split_for_day_model(df, day_ahead):
    """Split data for training a model that predicts day N ahead"""
    df['Date'] = pd.to_datetime(df['Date'])
    
    # For day N model, we need to exclude the last (7 + day_ahead) days from training
    # This ensures the model doesn't see data it would use for prediction
    buffer_days = 7 + day_ahead
    train_end_date = df['Date'].max() - pd.Timedelta(days=buffer_days)
    
    # Validation set: some recent data before the final prediction window
    val_start_date = train_end_date
    val_end_date = df['Date'].max() - pd.Timedelta(days=7)
    
    train = df[df['Date'] < train_end_date].copy()
    validation = df[(df['Date'] >= val_start_date) & (df['Date'] < val_end_date)].copy()
    
    # The prediction window for this specific day (24 hours)
    prediction_start_date = df['Date'].max() - pd.Timedelta(days=8-day_ahead)
    prediction_end_date = prediction_start_date + pd.Timedelta(days=1)
    
    day_prediction_window = df[(df['Date'] >= prediction_start_date) & 
                              (df['Date'] < prediction_end_date)].copy()
    
    train = train.drop(['Date'], axis=1, errors='ignore')
    validation = validation.drop(['Date'], axis=1, errors='ignore')
    day_prediction_window = day_prediction_window.drop(['Date'], axis=1, errors='ignore')
    
    print(f"Day {day_ahead} model - Train: {len(train)}, Validation: {len(validation)}, Day prediction window: {len(day_prediction_window)}")
    
    return train, validation, day_prediction_window

def train_all_day_models(df):
    """Train 7 models, each for predicting 1 day ahead"""
    models = {}
    feature_importances = {}
    day_windows = {}
    
    exclude_cols = ['Load', 'Date', 'Time Stamp', 'Load_1WeekAgo']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print("Training 7 day-ahead models...")
    print("=" * 50)
    
    for day in range(1, 8):  # Days 1-7
        print(f"\nTraining model for Day {day} prediction...")
        
        # Split data for this specific day model
        train, validation, day_prediction_window = split_for_day_model(df, day)
        
        # Store the day window for later use
        day_windows[f'day_{day}'] = day_prediction_window
        
        # Train the model
        model = train_day_model(train, feature_cols, validation, day)
        
        # Store the model
        models[f'day_{day}'] = model
        
        # Get feature importance
        fi = feature_importance(model, day)
        feature_importances[f'day_{day}'] = fi
        
        # Save individual model
        joblib.dump(model, f'xgboost_day_{day}_model.pkl')
        print(f"Saved Day {day} model to xgboost_day_{day}_model.pkl")
    
    return models, feature_importances, day_windows

def make_combined_forecast(df, models, day_windows):
    """Make 7-day forecast using the 7 trained models and their specific day windows"""
    exclude_cols = ['Load', 'Date', 'Time Stamp', 'Load_1WeekAgo']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    all_predictions = []
    daily_predictions = {}
    all_day_windows = []
    
    print("Making predictions with each day model...")
    
    for day in range(1, 8):
        # Get the specific day window that was created during training
        day_window = day_windows[f'day_{day}']
        
        if len(day_window) == 0:
            print(f"Warning: No data for Day {day}")
            continue
            
        # Get features for this day
        X_day = day_window[feature_cols]
        
        # Make prediction with the corresponding model
        model = models[f'day_{day}']
        day_pred = model.predict(X_day)
        
        # Store predictions
        daily_predictions[f'day_{day}'] = day_pred
        all_predictions.extend(day_pred)
        all_day_windows.append(day_window)
        
        print(f"Day {day}: Generated {len(day_pred)} hourly predictions")
    
    # Combine all day windows to create the full test prediction dataframe
    combined_test_prediction = pd.concat(all_day_windows, ignore_index=False)
    
    return all_predictions, daily_predictions, combined_test_prediction

def plot_combined_forecast(test_prediction, all_predictions, daily_predictions):
    """Plot the combined 7-day forecast"""
    # Plot overall forecast
    plt.figure(figsize=(20, 8))
    
    # Main plot - full 7 days
    plt.subplot(2, 1, 1)
    plt.plot(test_prediction.index, test_prediction['Load'], label='Actual Load', linewidth=2)
    plt.plot(test_prediction.index[:len(all_predictions)], all_predictions, 
             label='Combined Forecast', color='red', linewidth=2, linestyle='--')
    plt.title("7-Day Combined Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Daily breakdown
    plt.subplot(2, 1, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    start_idx = 0
    
    # Plot actual data
    plt.plot(test_prediction.index, test_prediction['Load'], 
             label='Actual Load', color='black', linewidth=2)
    
    # Plot each day's predictions with different colors
    for day in range(1, 8):
        if f'day_{day}' in daily_predictions:
            day_pred = daily_predictions[f'day_{day}']
            end_idx = start_idx + len(day_pred)
            
            day_indices = test_prediction.index[start_idx:end_idx]
            plt.plot(day_indices, day_pred, 
                    label=f'Day {day} Model', color=colors[day-1], 
                    linewidth=1, alpha=0.8)
            start_idx = end_idx
    
    plt.title("Individual Day Model Predictions")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_combined_forecast(all_predictions, test_prediction):
    """Evaluate the combined forecast performance"""
    actuals = test_prediction['Load'].values[:len(all_predictions)]
    predictions = np.array(all_predictions)
    
    # Calculate overall metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"\nCombined Forecast Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Calculate daily RMSE
    hours_per_day = 24
    num_days = min(len(predictions) // hours_per_day, 7)
    
    print(f"\nDaily RMSE breakdown:")
    daily_rmse = []
    for day in range(num_days):
        start_idx = day * hours_per_day
        end_idx = (day + 1) * hours_per_day
        
        daily_predictions = predictions[start_idx:end_idx]
        daily_actuals = actuals[start_idx:end_idx]
        
        if len(daily_predictions) > 0:
            daily_rmse_val = np.sqrt(mean_squared_error(daily_actuals, daily_predictions))
            daily_rmse.append(daily_rmse_val)
            print(f"Day {day + 1}: {daily_rmse_val:.2f}")
    
    return rmse, mae, mape, daily_rmse

def forecast_on_prediction_window(df, test_prediction, all_predictions):
    """Show forecast in context of full dataset"""
    test_prediction = test_prediction.copy()
    test_prediction['prediction'] = np.nan
    test_prediction.iloc[:len(all_predictions), test_prediction.columns.get_loc('prediction')] = all_predictions
    
    # Merge with full dataset to show context
    df_with_pred = df.copy()
    df_with_pred = df_with_pred.merge(test_prediction[['prediction']], 
                                     how='left', left_index=True, right_index=True)
    
    # Plot full dataset with predictions highlighted
    plt.figure(figsize=(20, 6))
    plt.plot(df_with_pred.index, df_with_pred['Load'], label='Historical Data', alpha=0.7)
    plt.plot(df_with_pred.index, df_with_pred['prediction'], 
             'o-', markersize=4, color='red', linewidth=2, label='7-Day Combined Forecast')
    plt.legend()
    plt.title('Full Dataset with 7-Day Combined Forecast Highlighted')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compare_feature_importance(feature_importances):
    """Compare feature importance across all day models"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE COMPARISON ACROSS DAY MODELS")
    print("="*60)
    
    # Get all unique features
    all_features = set()
    for day_fi in feature_importances.values():
        all_features.update(day_fi.index)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(index=sorted(all_features))
    
    for day, fi in feature_importances.items():
        comparison_df[day] = fi['importance']
    
    comparison_df = comparison_df.fillna(0)
    
    # Show top 10 features for each day
    print("\nTop 5 features for each day model:")
    for day in range(1, 8):
        day_col = f'day_{day}'
        if day_col in comparison_df.columns:
            top_features = comparison_df[day_col].nlargest(5)
            print(f"\nDay {day}:")
            for feature, importance in top_features.items():
                print(f"  {feature}: {importance:.4f}")
    
    return comparison_df

def main():
    path = '../data/processed/df.csv'
    df = pd.read_csv(path)
    
    print("Training 7 separate models for 7-day forecast...")
    print("Each model specializes in predicting 1 day ahead")
    print("=" * 60)
    
    # Train all 7 day models
    models, feature_importances, day_windows = train_all_day_models(df)
    
    print("\n" + "="*60)
    print("MAKING COMBINED 7-DAY FORECAST")
    print("="*60)
    
    # Make combined forecast
    all_predictions, daily_predictions, test_prediction = make_combined_forecast(df, models, day_windows)
    
    # Evaluate combined forecast
    rmse, mae, mape, daily_rmse = evaluate_combined_forecast(all_predictions, test_prediction)
    
    # Plot results
    print("\nPlotting combined forecast...")
    plot_combined_forecast(test_prediction, all_predictions, daily_predictions)
    
    print("\nShowing forecast in full dataset context...")
    forecast_on_prediction_window(df, test_prediction, all_predictions)
    
    # Compare feature importance across models
    comparison_df = compare_feature_importance(feature_importances)
    
    # Save feature importance comparison
    comparison_df.to_csv('feature_importance_comparison.csv')
    print(f"\nFeature importance comparison saved to feature_importance_comparison.csv")
    
    # Save all models as a single file
    all_models_data = {
        'models': models,
        'feature_importances': feature_importances,
        'daily_rmse': daily_rmse,
        'overall_metrics': {'rmse': rmse, 'mae': mae, 'mape': mape}
    }
    joblib.dump(all_models_data, 'xgboost_7day_combined_models.pkl')
    print("All models and metadata saved to xgboost_7day_combined_models.pkl")
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Trained 7 specialized day-ahead models")
    print(f"✓ Combined RMSE: {rmse:.2f}")
    print(f"✓ Combined MAE: {mae:.2f}")
    print(f"✓ Combined MAPE: {mape:.2f}%")
    print(f"✓ Models saved individually and as combined package")
    print(f"✓ Feature importance analysis completed")

if __name__ == "__main__":
    main()