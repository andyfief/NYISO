import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_forecast(predictions, actuals):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\nOverall Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    hours_per_day = 24
    num_days = len(predictions) // hours_per_day

    if num_days > 0:
        print(f"\nDaily RMSE progression:")
        for day in range(num_days):
            start_idx = day * hours_per_day
            end_idx = (day + 1) * hours_per_day
            daily_rmse = np.sqrt(mean_squared_error(
                actuals[start_idx:end_idx],
                predictions[start_idx:end_idx]
            ))
            print(f"Day {day + 1}: {daily_rmse:.2f}")

def main():
    # Load data
    df = pd.read_csv('../data/processed/df.csv', parse_dates=['Time Stamp'])
    df.set_index('Time Stamp', inplace=True)
    df = df.sort_index()

    # Define lengths
    forecast_horizon = 168  # 2 weeks (24*14)
    train_size = 5000       # training on 5,000 rows before last 2 weeks

    # Calculate indices
    train_start = len(df) - forecast_horizon - train_size  # start 5,000 rows before forecast window
    train_end = len(df) - forecast_horizon                # end before forecast window
    test_start = train_end
    test_end = len(df)

    # Subset data
    train = df.iloc[train_start:train_end]
    test = df.iloc[test_start:test_end]

    print(f"Training from {train.index[0]} to {train.index[-1]} ({len(train)} rows)")
    print(f"Testing from {test.index[0]} to {test.index[-1]} ({len(test)} rows)")

    # Load SARIMA model trained on these 5,000 rows
    print("Loading pmdarima model trained on 5k rows...")
    model = joblib.load('sarima_model_5k.pkl')

    # Forecast 336 steps (2 weeks)
    print(f"Forecasting next {forecast_horizon} steps...")
    forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

    # Plot forecast vs actual
    plt.figure(figsize=(15, 5))
    plt.plot(test.index, test['Load'], label='Actual Load')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title("pmdarima SARIMA: Forecast vs Actual (After Training Window)")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate
    evaluate_forecast(forecast, test['Load'].values)

if __name__ == "__main__":
    main()
