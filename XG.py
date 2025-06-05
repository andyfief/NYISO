import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def add_rolling_features(df, target='Load', window=3):
    df = df.copy()
    rolling_window = 2 * window + 1

    # Centered rolling average (look-ahead; only for training)
    df[f'{target}Rolling{window}'] = df[target].rolling(window=rolling_window, center=True, min_periods=1).mean()

    # Trailing rolling average (causal; usable at inference)
    df[f'{target}Trailing{window}'] = df[target].rolling(window=window, min_periods=1).mean()

    return df

def main():
    # Load data
    df = pd.read_csv('./data/final.csv')
    
    # Add rolling average features
    df = add_rolling_features(df, target='Load', window=3)

    # Define target and features
    target = 'Load'
    centered_feature = 'LoadRolling3'
    trailing_feature = 'LoadTrailing3'

    X = df.drop(columns=[target])
    y = df[target]

    # Keep only numeric columns
    X = X.select_dtypes(include=['number'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ⚠️ Drop centered (future-peeking) feature from test set only
    if centered_feature in X_test.columns:
        X_test[centered_feature] = float('nan')

    # Train the model (centered + trailing available in training)
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
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on test set: {mse:.4f}')
    print(f'Off by {np.sqrt(mse)}')

    # Plot predicted vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[100:200], label='Actual Load', alpha=0.7)
    plt.plot(y_pred[100:200], label='Predicted Load', alpha=0.7)
    plt.title('XGBoost Load Prediction vs Actual (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Load')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
