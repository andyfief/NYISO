from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily, Hourly

# Set time period (reduced to more manageable range)
start = datetime(2005, 1, 31)
end = datetime(2025, 4, 26)

# Central Park coordinates (note: longitude should be negative for NYC)
CentralPark = Point(40.7128, -74.0060)  # Fixed longitude sign

# Get hourly data
data = Hourly(CentralPark, start, end)
df = data.fetch()  # Assign to df variable to match your print statement

# Show sample data
print(df.head())

# Optional: Show data info
print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")

df = df.reset_index()

savefile = 'weatherDF.csv'
df.to_csv(savefile, index=False)
print(f"Data saved to {savefile}")


