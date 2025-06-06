import pandas as pd
import pytz
from datetime import datetime, timedelta
import calendar
import csv

def clean_null_load_values(df):
    """Removes 3 null load values, converts Time Stamp and timezone to UTC, drops both old ones"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()

    df = df.dropna(subset=['Integrated Load'])
    df = df.drop('Name', axis=1)
    
    df = df.drop('PTID', axis=1)
    df = df.rename(columns={'Integrated Load': 'Load'})
    df = df.rename(columns={'UTC_Time Stamp': 'Time Stamp'})

    return df

def convert_to_utc(df):
    df = df.copy()
    eastern = pytz.timezone('US/Eastern')

    def convert(row):
        if isinstance(row['Time Stamp'], str):
            dt = datetime.strptime(row['Time Stamp'], '%m/%d/%Y %H:%M:%S')
        else:
            dt = row['Time Stamp']  # already a datetime

        is_dst = row['Time Zone'] == 'EDT'
        localized = eastern.localize(dt, is_dst=is_dst)
        return localized.astimezone(pytz.UTC)

    df['Time Stamp'] = df.apply(convert, axis=1)  # keep datetime
    
    df = df.drop('Time Zone', axis=1)
    #df = df.drop('Time Stamp', axis=1)
    return df

def hourDay(df):
    df = df.copy()

    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])

    df['Date'] = df['Time Stamp'].dt.date
    df['Hour'] = df['Time Stamp'].dt.hour

    return df

def add_time_features(df):
    """Adds time features: Day, Hour, Month, Season, IsWeekend"""
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
    Adds day-based rolling average columns for the target column.
    Uses actual calendar days, not row indexes.
    """
    df = df.copy()
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Keep the simple trailing averages (row-based) for immediate context
    df[f'{target}Trailing{1}'] = df[target].rolling(window=1, min_periods=1).mean()
    df[f'{target}Trailing{3}'] = df[target].rolling(window=3, min_periods=1).mean()
    df[f'{target}Trailing{7}'] = df[target].rolling(window=7, min_periods=1).mean()
    
    # Add day-based rolling averages
    # First, get daily averages for the target
    daily_avg = df.groupby('Date')[target].mean()
    
    # Create day-based rolling averages
    daily_rolling_3 = daily_avg.rolling(window=3, min_periods=1).mean()
    daily_rolling_7 = daily_avg.rolling(window=7, min_periods=1).mean()
    daily_rolling_14 = daily_avg.rolling(window=14, min_periods=1).mean()
    
    # Map back to original dataframe
    df[f'{target}DayRolling3'] = df['Date'].map(daily_rolling_3)
    df[f'{target}DayRolling7'] = df['Date'].map(daily_rolling_7)
    df[f'{target}DayRolling14'] = df['Date'].map(daily_rolling_14)
    
    return df

def lag_average(df, newColumn, days, target='Load'):
    """
    Adds a new column with the average of 'target' column N days ago.
    """
    df = df.copy()

    # Ensure 'Date' column is datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    # Group by Date and calculate daily mean for target (in case multiple rows per day)
    daily_avg = df.groupby('Date')[target].mean()

    # Shift by 'days' to get lag
    lagged_avg = daily_avg.shift(days)

    # Map lagged average back to each row by date
    df[f'{str(newColumn)}'] = df['Date'].map(lagged_avg)

    return df

def numericalDate(df):
    df = df.copy()
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])

    df['Year'] = df['Time Stamp'].dt.year
    df['DayOfYear'] = df['Time Stamp'].dt.dayofyear
    df['DayOfWeek'] = df['Time Stamp'].dt.weekday + 1  # Monday=1, Sunday=7

    season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
    df['seasonNum'] = df['Season'].map(season_mapping)

    df = df.drop('Season', axis = 1)
    df = df.drop('Day', axis=1)
    #df = df.drop('Date', axis=1)

    return df

def add_temperature(df, weathercsv):
    """Merge temperature from weather data into the main DataFrame using UTC Time Stamps."""
    df = df.copy()

    # Load weather data
    weather_df = pd.read_csv(weathercsv)

    # Ensure datetime format
    weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], utc=True)

    # Rename to align with main DataFrame
    weather_df = weather_df.rename(columns={'time': 'Time Stamp', 'temp': 'Temperature'})

    # Merge on exact Time Stamp
    df = pd.merge(df, weather_df[['Time Stamp', 'Temperature']], on='Time Stamp', how='left')

    return df

def twelveHourTemp(df):
    """
    Create 12-hour average temperature feature for 9AM-9PM blocks.
    Drops all rows before the first 9:00AM Time Stamp.
    """
    df = df.copy()
    
    # Ensure we have the required columns
    if 'Time Stamp' not in df.columns:
        raise ValueError("Time Stamp column is required for twelveHourTemp function")
    
    # Ensure Time Stamp is datetime and in UTC
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], utc=True)
    
    # Extract hour and date for processing
    df['TempHour'] = df['Time Stamp'].dt.hour
    df['TempDate'] = df['Time Stamp'].dt.date
    
    # Find the first occurrence of 9AM and drop everything before it
    first_9am_idx = None
    for idx, row in df.iterrows():
        if row['TempHour'] == 9:
            first_9am_idx = idx
            break
    
    if first_9am_idx is None:
        raise ValueError("No 9AM Time Stamp found in data - cannot proceed")
    
    # Drop all rows before first 9AM
    df = df.iloc[first_9am_idx:].reset_index(drop=True)
    
    # Recalculate TempHour and TempDate after dropping rows
    df['TempHour'] = df['Time Stamp'].dt.hour
    df['TempDate'] = df['Time Stamp'].dt.date
    
    # Group by date and calculate 9AM-9PM averages
    daily_averages = {}
    
    for date in df['TempDate'].unique():
        date_data = df[df['TempDate'] == date]
        
        # Filter for 9AM to 9PM (inclusive of 9PM = hour 21)
        daytime_data = date_data[
            (date_data['TempHour'] >= 9) & (date_data['TempHour'] <= 21)
        ]
        
        if len(daytime_data) > 0:
            daily_averages[date] = daytime_data['Temperature'].mean()
    
    # Assign daily averages to all rows for each date
    df['averageTemp'] = df['TempDate'].map(daily_averages)
    
    # Clean up temporary columns
    df = df.drop(columns=['TempHour', 'TempDate'])
    
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

    print(df.dtypes)

    print("Ordering by Time Stamp...")
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], format='%m/%d/%Y %H:%M:%S')
    df = df.sort_values('Time Stamp').reset_index(drop=True)

    print("Converting to UTC...")
    df = convert_to_utc(df)

    # Step 1: Clean null values and convert to UTC
    print("Cleaning Nulls")
    df = clean_null_load_values(df)

    print("Adding temp from weather csv...")
    df = add_temperature(df, 'weatherDF.csv')

    print("Adding average temperature for every 12 hours...")
    df = twelveHourTemp(df)

    print("Removing Time Stamp and converting to Hour, Day")
    df = hourDay(df)

    print('Adding time features...')
    df = add_time_features(df)

    print("Adding rolling averages...")
    df = rollingAverages(df)

    print("Adding 1 day lag...")
    df = lag_average(df, '1DayLag', 1)

    print("Using numerical DOTW, DOTY and Year....")
    df = numericalDate(df)

    #print("Dropping Time Stamp...")
    #df = df.drop('Time Stamp', axis=1)

    print("Saving...")
    save_to_csv(df, 'final.csv')

if __name__ == "__main__":
    main()