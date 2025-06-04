import pandas as pd
import pytz
from datetime import datetime
import holidays
import calendar

def clean_null_load_values(df):
    """Removes 3 null load values, converts timestamp and timezone to UTC, drops both old ones"""
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Dropped 3 null values for load
    null_load_rows = df[df['Integrated Load'].isnull()]
    print(null_load_rows)
    df = df.dropna(subset=['Integrated Load']).copy()  # Add .copy() to ensure we have a proper dataframe

    unique_timezones = df['Time Zone'].unique()
    print(unique_timezones)

    def convert_to_utc(row):
        timestamp_str = row['Time Stamp']  # '01/31/2005 00:00:00'
        timezone = row['Time Zone']  # 'EST' or 'EDT'
        
        dt = datetime.strptime(timestamp_str, '%m/%d/%Y %H:%M:%S')
        
        # Create timezone aware datetime
        # Both EST and EDT are handled by the US/Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        
        #use the timezone name to determine if it's standard time or DST
        if timezone == 'EST':
            # Force interpret as standard time
            dt = eastern.localize(dt, is_dst=False)
        else:  # EDT
            # Force interpret as daylight saving time
            dt = eastern.localize(dt, is_dst=True)
        
        # Convert to UTC
        utc_dt = dt.astimezone(pytz.UTC)
        
        return utc_dt

    # Apply the function to create a new UTC timestamp column
    df['UTC_Timestamp'] = df.apply(convert_to_utc, axis=1)
    df['UTC_Timestamp'] = df['UTC_Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    #drop old
    df = df.drop('Time Zone', axis=1)
    df = df.drop('Time Stamp', axis=1)

    return df

def rename_columns_and_drop_ptid(df):
    """Renames some columns and drops PTID"""
    df = df.copy()
    
    df = df.drop('PTID', axis=1)
    df = df.rename(columns={'Integrated Load': 'Load'})
    df = df.rename(columns={'UTC_Timestamp': 'Timestamp'})

    return df

def add_time_features(df):
    """Adds time features: Day, Hour, Month, Season, IsWeekend. Removes Timestamp."""
    df = df.copy()
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create separate date and hour columns
    df['Date'] = df['Timestamp'].dt.date
    df['Hour'] = df['Timestamp'].dt.hour
    df = df.drop('Timestamp', axis=1)

    df['Day'] = pd.to_datetime(df['Date']).dt.day_name()
    df['IsWeekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)

    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Season'] = pd.to_datetime(df['Date']).dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring',
        5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'
    })

    return df

def calculate_thanksgiving(year):
    """Calculate Thanksgiving date for a given year"""
    # Get the calendar for November of the given year
    c = calendar.monthcalendar(year, 11)
    
    # The fourth Thursday is the Thursday (3rd day of week with Monday=0) 
    # in the fourth week that contains a Thursday
    thanksgiving_day = 0
    thursday_count = 0
    
    for week in c:
        # Thursday has index 3 (0-based, with Monday as 0)
        if week[3] != 0:
            thursday_count += 1
            if thursday_count == 4:
                thanksgiving_day = week[3]
                break
    
    return f"{year}-11-{thanksgiving_day:02d}"

def add_holiday_features(df):
    """Add holiday names, isHoliday, isBeforeHoliday, isAfterHoliday"""
    # add holiday names, isHoliday, isBeforeHoliday, isAfterHoliday --- needs work
    """
    DONE:
    New Year's Day
    New Year's Day (observed)
    Martin Luther King Jr. Day
    Washington's Birthday
    Memorial Day
    Juneteenth National Independence Day
    Independence Day
    Labor Day
    Columbus Day
    Veterans Day (observed)
    Veterans Day
    Thanksgiving Day
    Christmas Day
    New Years Eve YYYY-12-31
    Christmas Eve YYYY-12-24
    Halloween YYYY-10-31
    Black Friday
    Cyber Monday ----------- The first cyberMonday was in 2005 which is convinient but i might omit this anyway
    Easter Sunday

    ADD:
    Pride Weekend
    NYC Marathon
    Five Boro Bike Tour
    Macy's Thanksgiving Day Parade
    St Patricks Day Parade
    Puerto Rican Day Parade
    Domincan Day Parade
    West Indian Day Parade
    Tribeca Film Festival
    New York Fashion Week
    UN General Assembly Week
    Christmas Tree Lighting at Rockefeller Center
    Christmas Break
    Spring break
    Summer break
    First day of school?
    last day of school?
        https://web.archive.org/
        schools.nyc.gov/calendar

    Super bowl
        02/06/2005
        02/05/2006
        02/04/2007
        02/03/2008
        02/01/2009
        02/07/2010
        02/06/2011
        02/05/2012
        02/03/2013
        02/02/2014
        02/01/2015
        02/07/2016
        02/05/2017
        02/04/2018
        02/03/2019
        02/02/2020
        02/07/2021
        02/13/2022
        02/12/2023
        02/11/2024
        02/09/2025
    Oscars
    Presedential Debates
    Election days
    Yankees/Mets home games
    Broadway shutdowns/openings

    Covid lockdowns, by level/restriction
    """
    
    def add_federal_holidays(df):
        """Add federal holidays from holidays library"""
        start_year = df['Date'].min().year
        end_year = df['Date'].max().year
        us_holidays = holidays.US(years=range(start_year, end_year + 1))

        holidays_list = {
            "newYearsDay": "New Year's Day",
            "newYearsDayObserved": "New Year's Day (observed)",
            "martinLutherKingJrDay": "Martin Luther King Jr. Day",
            "washingtonsBirthday": "Washington's Birthday",
            "memorialDay": "Memorial Day",
            "juneteenthNationalIndependenceDay": "Juneteenth National Independence Day",
            "independenceDay": "Independence Day",
            "laborDay": "Labor Day",
            "columbusDay": "Columbus Day",
            "veteransDayObserved": "Veterans Day (observed)",
            "veteransDay": "Veterans Day",
            "thanksgivingDay": "Thanksgiving Day",
            "christmasDay": "Christmas Day"
        }

        for holiday in holidays_list:
            holidayTag = holidays_list[holiday]
            df[holiday] = df['Date'].apply(
                lambda x: 1 if us_holidays.get(x) == holidayTag else 0
            )
        return df

    def add_fixed_date_holidays(df):
        """Add holidays that occur on fixed dates each year"""
        datesToAdd = {
            "newYearsEve": ("12", "31"),
            "christmasEve": ("12", "24"),
            "valentinesDay": ("02", "14"),
            "halloween": ("10", "31"),
        }

        # Loop through each special day
        for holiday_name, (month, day) in datesToAdd.items():
            df[holiday_name] = df['Date'].apply(
                lambda x: 1 if (x.month == int(month) and x.day == int(day)) else 0
            )
        return df

    def add_thanksgiving_related_holidays(df):
        """Add Thanksgiving, Black Friday, and Cyber Monday"""
        # Calculate Thanksgiving for all years from 2005 to current year
        current_year = datetime.now().year
        thanksgiving_dates = []

        for year in range(2005, current_year):
            thanksgiving_dates.append(calculate_thanksgiving(year))

        thanksgiving_dates = pd.to_datetime(thanksgiving_dates)

        blackfriday_dates = []
        for date in thanksgiving_dates:
            blackfriday_dates.append(date + pd.Timedelta(days=1))

        blackfriday_dates = pd.to_datetime(blackfriday_dates)

        cybermonday_dates = []
        for date in thanksgiving_dates:
            cybermonday_dates.append(date + pd.Timedelta(days=4))

        cybermonday_dates = pd.to_datetime(cybermonday_dates)

        df['thanksgiving'] = 0
        for date in thanksgiving_dates:
            df.loc[df['Date'] == date, 'thanksgiving'] = 1
        df['blackFriday'] = 0
        for date in blackfriday_dates:
            df.loc[df['Date'] == date, 'blackFriday'] = 1
        df['cyberMonday'] = 0
        for date in cybermonday_dates:
            df.loc[df['Date'] == date, 'cyberMonday'] = 1

        return df

    def add_easter_holiday(df):
        """Add Easter Sunday dates"""
        easter_dates = ['2005-03-27', '2006-04-16', '2007-04-08', '2008-03-23', 
                        '2009-04-12', '2010-04-04', '2011-04-24', '2012-04-08', 
                        '2013-03-31', '2014-04-20', '2015-04-05', '2016-03-27', 
                        '2017-04-16', '2018-04-01', '2019-04-21', '2020-04-12', 
                        '2021-04-04', '2022-04-17', '2023-04-09', '2024-03-31', 
                        '2025-04-20']
        
        easter_dates = pd.to_datetime(easter_dates)
        df['easter'] = 0
        for date in easter_dates:
            df.loc[df['Date'] == date, 'easter'] = 1

        return df

    def add_school_years_and_breaks(df):
        """Add school-related dates (first day, last day, breaks)"""
        # TODO: Implement school calendar dates
        # Will need to research NYC school calendar archives
        return df

    def add_sporting_events(df):
        """Add major sporting events (Super Bowl, Yankees/Mets games, etc.)"""
        # Super Bowl dates
        superbowl_dates = ['2005-02-06', '2006-02-05', '2007-02-04', '2008-02-03', 
                          '2009-02-01', '2010-02-07', '2011-02-06', '2012-02-05', 
                          '2013-02-03', '2014-02-02', '2015-02-01', '2016-02-07', 
                          '2017-02-05', '2018-02-04', '2019-02-03', '2020-02-02', 
                          '2021-02-07', '2022-02-13', '2023-02-12', '2024-02-11', 
                          '2025-02-09']
        
        superbowl_dates = pd.to_datetime(superbowl_dates)
        df['superBowl'] = 0
        for date in superbowl_dates:
            df.loc[df['Date'] == date, 'superBowl'] = 1

        # TODO: Add Yankees/Mets home games
        # TODO: Add NYC Marathon, other major sporting events
        return df

    def add_parades_and_festivals(df):
        """Add NYC parades and festivals"""
        # TODO: Implement dates for:
        # - Pride Weekend
        # - Five Boro Bike Tour
        # - Macy's Thanksgiving Day Parade
        # - St Patricks Day Parade
        # - Puerto Rican Day Parade
        # - Dominican Day Parade
        # - West Indian Day Parade
        # - Tribeca Film Festival
        # - New York Fashion Week
        # - UN General Assembly Week
        # - Christmas Tree Lighting at Rockefeller Center
        return df

    def add_presidential_debates(df):
        """Add presidential debate dates"""
        # TODO: Implement presidential debate dates
        return df

    def add_oscars(df):
        """Add Oscar ceremony dates"""
        # TODO: Implement Oscar ceremony dates
        return df

    def add_elections(df):
        """Add election days"""
        # TODO: Implement election dates (federal, state, local)
        return df

    def add_broadway_events(df):
        """Add Broadway shutdowns/openings"""
        # TODO: Implement Broadway-related dates
        return df

    def add_covid_restrictions(df):
        """Add Covid lockdowns by level/restriction"""
        # TODO: Implement Covid restriction periods
        return df

    # Main function execution
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Add all holiday groups
    df = add_federal_holidays(df)
    df = add_fixed_date_holidays(df)
    df = add_thanksgiving_related_holidays(df)
    df = add_easter_holiday(df)
    df = add_school_years_and_breaks(df)
    df = add_sporting_events(df)
    df = add_parades_and_festivals(df)
    df = add_presidential_debates(df)
    df = add_oscars(df)
    df = add_elections(df)
    df = add_broadway_events(df)
    df = add_covid_restrictions(df)

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
    print("Cleaning Nulls and Converting to UTC...")
    df = clean_null_load_values(df)
    
    # Step 2: Rename columns and drop PTID
    print("Renaming some columns...")
    df = rename_columns_and_drop_ptid(df)
    
    # Step 3: Add time features
    print("Adding time features...")
    df = add_time_features(df)
    
    # Step 4: Add holiday features
    print("Adding holiday features...")
    df = add_holiday_features(df)

    print("Saving to new csv...")
    save_to_csv(df, "final_processed_data.csv") 

    print("bye")

if __name__ == "__main__":
    main()