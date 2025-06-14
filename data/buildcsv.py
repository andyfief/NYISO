import pandas as pd
import pytz
from datetime import datetime, timedelta
import calendar
import csv

def clean_null_load_values(df):
    """Removes 3 null load values, converts timestamp and timezone to UTC, drops both old ones"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    df = df.dropna(subset=['Integrated Load'])
    df = df.drop('Name', axis=1)
    
    df = df.drop('PTID', axis=1)
    df = df.rename(columns={'Integrated Load': 'Load'})
    df = df.rename(columns={'UTC_Timestamp': 'Timestamp'})

    return df

def convert_to_utc(df):
    """Converts 'Time Stamp' and 'Time Zone' columns to UTC datetime and cleans up the DataFrame."""
    df = df.copy()

    eastern = pytz.timezone('US/Eastern')

    def convert(row):
        dt = datetime.strptime(row['Time Stamp'], '%m/%d/%Y %H:%M:%S')
        is_dst = row['Time Zone'] == 'EDT'
        localized = eastern.localize(dt, is_dst=is_dst)
        return localized.astimezone(pytz.UTC)

    df['UTC_Timestamp'] = df.apply(convert, axis=1)
    df['UTC_Timestamp'] = df['UTC_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df = df.drop(columns=['Time Stamp', 'Time Zone'])

    df = df.rename(columns={'UTC_Timestamp': 'Timestamp'})

    return df

def hourDay(df):
    df = df.copy()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour

    return df

def add_time_features(df):
    """Adds time features: Day, Hour, Month, Season, IsWeekend. Removes Timestamp."""
    df = df.copy()

    df['Day'] = pd.to_datetime(df['Date']).dt.day_name()

    df['IsWeekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)

    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Season'] = pd.to_datetime(df['Date']).dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    })

    return df

def rollingAverages(df, target='Load'):
    """
    Adds a trailing (causal) rolling average column for the target column.
    window: number of past frames to include (does not peek into the future).
    """
    df = df.copy()

    df[f'{target}Trailing{1}'] = df[target].rolling(window=1, min_periods=1).mean()
    df[f'{target}Trailing{2}'] = df[target].rolling(window=2, min_periods=2).mean()
    df[f'{target}Trailing{3}'] = df[target].rolling(window=3, min_periods=1).mean()
    df[f'{target}Trailing{7}'] = df[target].rolling(window=7, min_periods=1).mean()

    df[f'{target}Centered{3}'] = df[target].rolling(window=3, center=True, min_periods=1).mean()
    df[f'{target}Centered{7}'] = df[target].rolling(window=7, center=True, min_periods=1).mean()

    return df

def lag_average(df, newColumn, days, target='Load'):
    """
    Adds a new column with the average of 'target' column N days ago.
    """
    df = df.copy()

    # Ensure 'Date' column is datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date to make sure lag makes sense
    df = df.sort_values('Date')

    # Group by Date and calculate daily mean for target (in case multiple rows per day)
    daily_avg = df.groupby('Date')[target].mean()

    # Shift by 'days' to get lag
    lagged_avg = daily_avg.shift(days)

    # Map lagged average back to each row by date
    df[f'{str(newColumn)}'] = df['Date'].map(lagged_avg)

    return df

def numericalDate(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['Year'] = df['Timestamp'].dt.year
    df['DayOfYear'] = df['Timestamp'].dt.dayofyear
    df['DayOfWeek'] = df['Timestamp'].dt.weekday + 1  # Monday=1, Sunday=7

    season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
    df['seasonNum'] = df['Season'].map(season_mapping)

    df = df.drop('Season', axis = 1)
    df = df.drop('Day', axis=1)
    #df = df.drop('Date', axis=1)

    return df

def add_temperature(df, weathercsv):
    """Merge temperature from weather data into the main DataFrame using UTC timestamps."""
    df = df.copy()

    # Load weather data
    weather_df = pd.read_csv(weathercsv)

    # Ensure datetime format
    weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

    # Rename to align with main DataFrame
    weather_df = weather_df.rename(columns={'time': 'Timestamp', 'temp': 'Temperature'})

    # Merge on exact timestamp
    df = pd.merge(df, weather_df[['Timestamp', 'Temperature']], on='Timestamp', how='left')

    return df

def hotColdThreshold(df):
    df = df.copy()
    df['hotFlag'] = 0
    df['coldFlag'] = 0
    #Found using grid search
    df.loc[df['Temperature'] > 20, 'hotFlag'] = 1
    df.loc[df['Temperature'] < 9, 'coldFlag'] = 1
    return df

def save_to_csv(df, filename):
    """Save dataframe to CSV file"""
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return df

def main():
    csv_path = "nyc_load_aggregated_raw.csv"

    print("Loading original csv file...")
    df = pd.read_csv(csv_path)

    # Step 1: Clean null values and convert to UTC
    print("Cleaning Nulls")
    df = clean_null_load_values(df)

    print("Converting to UTC...")
    df = convert_to_utc(df)

    print("Adding temp from weather csv...")
    df = add_temperature(df, 'weatherDF.csv')

    print("Adding weather threshold at 20, 9C")
    df = hotColdThreshold(df)

    print("Removing timestamp and converting to Hour, Day")
    df = hourDay(df)

    print('Adding time features...')
    df = add_time_features(df)

    print("Adding rolling averages...")
    df = rollingAverages(df)

    print("Adding 1 day lag...")
    df = lag_average(df, '1DayLag', 1)


    print("Using numerical DOTW, DOTY and Year....")
    df = numericalDate(df)

    print("Dropping Timestamp...")
    df = df.drop('Timestamp', axis=1)

    print("Saving...")
    save_to_csv(df, 'final.csv')

if __name__ == "__main__":
    main()