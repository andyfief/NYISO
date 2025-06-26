# A Virtual Environment running python 3.11 with conda is used to run this script due to pmdarima compatibility issues.
# With current number of models ran, 4000 is close to maximum rows without my PC having memory allocation errors.

import pandas as pd
from pmdarima import auto_arima

# Example time series data
# Replace this with your actual time series (e.g., hourly or daily energy load)
df = pd.read_csv("../../data/processed/df.csv", parse_dates=["Date"], index_col="Time Stamp")
middle = df.shape[0] // 2
half_window = 2000
df_middle = df.iloc[middle - half_window : middle + half_window]

y = df["Load"]  # or whatever your column is
y_middle = y.iloc[middle - half_window : middle + half_window]


# Fit SARIMA model automatically
model = auto_arima(
    y_middle,
    seasonal=True,
    m=24,                   # adjust for your seasonality
    max_p=2,
    max_q=2,
    max_P=1,
    max_Q=1,
    d=1,
    D=1,
    max_order=5,
    with_intercept=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    trace=True #ARIMA(2,1,0)(1,1,0)[24]
)
print(model.summary())

#conda activate arima-env