import pandas as pd
from pmdarima import auto_arima
import joblib

# Load full data
df = pd.read_csv('../data/processed/df.csv', parse_dates=['Date'], index_col='Time Stamp')
df = df.sort_index()

end = len(df) - 168      # exclude last 168 rows used for testing
start = end - 5000       # go back 5000 rows from there
df_subset = df.iloc[start:end]

# Extract series to fit
y = df_subset['Load']

# Fit auto_arima with your best model orders as starting point / constraints
model = auto_arima(y, 
                   start_p=2, max_p=2,
                   start_q=0, max_q=0,
                   d=1,
                   seasonal=True,
                   start_P=1, max_P=1,
                   start_Q=0, max_Q=0,
                   D=1,
                   m=24,
                   stepwise=True,
                   suppress_warnings=True,
                   error_action='ignore')

# Save the fitted model
joblib.dump(model, 'sarima_model_5k.pkl')
print("Model saved to sarima_model_5k.pkl")