import pandas as pd
import pytz
from datetime import datetime, timedelta
import calendar
import csv
import numpy as np
from typing import List, Tuple, Dict

def clean_null_load_values(df):
    """Removes null load values, drops unnecessary columns, renames columns"""
    df = df.copy()
    df = df.dropna(subset=['Integrated Load'])
    df = df.drop('Name', axis=1, errors='ignore')
    df = df.drop('PTID', axis=1, errors='ignore')
    df = df.rename(columns={'Integrated Load': 'Load'})
    df = df.rename(columns={'UTC_Time Stamp': 'Time Stamp'})

    return df

def reindex_interpolate(df):
    df = df.copy()
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'])
    df = df.set_index('Time Stamp')
    df = df.sort_index()

    df = df[~df.index.duplicated(keep='first')] # keep the first timestamp if there are duplicates

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h') # full expected range of hours
    df = df.reindex(full_index) # put data back into full range
    df.index.name = 'Time Stamp'

    df['Load'] = df['Load'].interpolate(method='linear') # interpolate missing values in full range

    return df

def convert_to_utc(df):
    """Convert timezone-aware timestamps to UTC"""
    df = df.copy()
    eastern = pytz.timezone('US/Eastern')

    def convert(row):
        if isinstance(row['Time Stamp'], str):
            dt = datetime.strptime(row['Time Stamp'], '%m/%d/%Y %H:%M:%S')
        else:
            dt = row['Time Stamp']
        
        is_dst = row['Time Zone'] == 'EDT'
        localized = eastern.localize(dt, is_dst=is_dst)
        return localized.astimezone(pytz.UTC)

    df['Time Stamp'] = df.apply(convert, axis=1)
    df = df.drop('Time Zone', axis=1)
    return df

def add_temperature(df, weathercsv):
    """Merge temperature data using UTC timestamps"""
    df = df.copy()
    
    # Load and process weather data
    weather_df = pd.read_csv(weathercsv)
    weather_df['time'] = pd.to_datetime(weather_df['time'], utc=True)
    
    # Reset index to make Time Stamp a column for merging
    df_for_merge = df.reset_index()
    df_for_merge['Time Stamp'] = pd.to_datetime(df_for_merge['Time Stamp'], utc=True)
    
    # Rename and merge
    weather_df = weather_df.rename(columns={'time': 'Time Stamp', 'temp': 'Temperature'})
    df_merged = pd.merge(df_for_merge, weather_df[['Time Stamp', 'Temperature']], on='Time Stamp', how='left')
    
    # Set Time Stamp back as index
    df_merged = df_merged.set_index('Time Stamp')
    
    return df_merged

def twelveHourTemp(df):
    """Create 12-hour average temperature feature for 9AM-9PM blocks"""
    df = df.copy()
    
    # Reset index to work with Time Stamp as column
    df_temp = df.reset_index()
    
    if 'Time Stamp' not in df_temp.columns:
        raise ValueError("Time Stamp column is required")
    
    df_temp['Time Stamp'] = pd.to_datetime(df_temp['Time Stamp'], utc=True)
    df_temp['TempHour'] = df_temp['Time Stamp'].dt.hour
    df_temp['TempDate'] = df_temp['Time Stamp'].dt.date
    
    # Find first 9AM and drop everything before it
    first_9am_idx = None
    for idx, row in df_temp.iterrows():
        if row['TempHour'] == 9:
            first_9am_idx = idx
            break
    
    if first_9am_idx is None:
        raise ValueError("No 9AM timestamp found in data")
    
    df_temp = df_temp.iloc[first_9am_idx:].reset_index(drop=True)
    df_temp['TempHour'] = df_temp['Time Stamp'].dt.hour
    df_temp['TempDate'] = df_temp['Time Stamp'].dt.date
    
    # Calculate 9AM-9PM daily averages
    daily_averages = {}
    for date in df_temp['TempDate'].unique():
        date_data = df_temp[df_temp['TempDate'] == date]
        daytime_data = date_data[(date_data['TempHour'] >= 9) & (date_data['TempHour'] <= 21)]
        
        if len(daytime_data) > 0:
            daily_averages[date] = daytime_data['Temperature'].mean()
    
    df_temp['averageTemp'] = df_temp['TempDate'].map(daily_averages)
    df_temp = df_temp.drop(columns=['TempHour', 'TempDate'])
    
    # Set Time Stamp back as index
    df_temp = df_temp.set_index('Time Stamp')
    
    return df_temp

def create_time_features(df):
    df = df.copy()

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Create Date column from index for season mapping
    df['Date'] = df.index.date
    df['Season'] = pd.to_datetime(df['Date']).dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    })
    season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
    df['seasonNum'] = df['Season'].map(season_mapping)

    df = df.drop(['Season'], axis=1, errors='ignore')

    return df

def addLag(df):
    df = df.copy()

    # Step 1: Create lag reference dates
    df['DateMinus1Day'] = df['Date'] - pd.Timedelta(days=1)
    df['Date1WeekAgo'] = df['Date'] - pd.Timedelta(days=7)

    # Step 2: Make a simple lookup table for Date -> Load
    load_lookup = df[['Date', 'Load']].copy()

    # Step 3: Merge Load_1DayAgo
    df = df.merge(load_lookup.rename(columns={'Date': 'DateMinus1Day', 'Load': 'Load_1DayAgo'}),
                  on='DateMinus1Day', how='left')

    # Step 4: Merge Load_1WeekAgo
    df = df.merge(load_lookup.rename(columns={'Date': 'Date1WeekAgo', 'Load': 'Load_1WeekAgo'}),
                  on='Date1WeekAgo', how='left')

    # Step 5: Drop the lag keys if desired
    df.drop(columns=['DateMinus1Day', 'Date1WeekAgo'], inplace=True)

    return df

def save_to_csv(df, filename):
    df.to_csv(filename, index=True)  # Keep index since it's Time Stamp
    print(f"Data saved to {filename}")
    return df

def main():
    csv_path = "../data/raw/nyc_load_aggregated_raw.csv"
    weather_file = "../data/raw/weatherDF.csv"

    df = pd.read_csv(csv_path)
    print("Cleaning Null Values...")
    df = clean_null_load_values(df)
    print("Converting to UTC...")
    df = convert_to_utc(df)  # This needs Time Stamp as a column
    print("Reindexing, Interpolating...")
    df = reindex_interpolate(df)  # This sets Time Stamp as index
    print("Creating time features...")
    df = create_time_features(df)  # This needs Time Stamp as index
    print("Adding temperature data...")
    df = add_temperature(df, weather_file)  # Modified to handle index properly
    print("Adding trailing average of temperature...")
    df = twelveHourTemp(df)  # Modified to handle index properly
    print("Adding lag...")
    df = addLag(df)

    save_to_csv(df, '../data/processed/df.csv') # For comparing to actuals in XG. I could put everything below this in XG too
    

if __name__ == "__main__":
    main()