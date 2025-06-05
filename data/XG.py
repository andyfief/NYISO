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
    df = pd.read_csv('final.csv')

    df = df.drop('Temperature', axis=1)
    
    # Split first to avoid data leakage
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Add rolling features to each split separately
    train_df = add_rolling_features(train_df, target='Load', window=3)
    test_df = add_rolling_features(test_df, target='Load', window=3)
    
    # Remove centered (future-peeking) feature from test set
    if 'LoadRolling3' in test_df.columns:
        test_df['LoadRolling3'] = float('nan')
    
    # Prepare features and target
    X_train = train_df.drop(columns=['Load']).select_dtypes(include=['number'])
    y_train = train_df['Load']
    X_test = test_df.drop(columns=['Load']).select_dtypes(include=['number'])
    y_test = test_df['Load']

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

    # Plot predicted vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[100:200], label='Actual Load', alpha=0.7)
    plt.plot(y_pred_adj[100:200], label='Predicted Load', alpha=0.7)
    plt.title('XGBoost Load Prediction vs Actual (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Load')
    plt.legend()
    plt.show()

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

    # Plot it
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
