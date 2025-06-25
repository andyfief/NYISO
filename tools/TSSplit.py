
def create_time_series_splits(df: pd.DataFrame, 
                            n_splits: int = 6, 
                            test_size_days: int = 7,
                            min_train_days: int = 60) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create time series cross-validation splits with growing windows.
    
    Args:
        df: DataFrame with 'Date' column
        n_splits: Number of CV splits to create
        test_size_days: Size of each test period in days
        min_train_days: Minimum training period in days
    
    Returns:
        List of (train_end_date, test_end_date) tuples
    """
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get data date range
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    total_days = (end_date - start_date).days
    
    print(f"Data spans from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_days} days)")
    
    # Calculate split dates working backwards from end
    splits = []
    
    # Space test periods evenly across the latter portion of the data
    # Leave enough space for minimum training period
    available_days = total_days - min_train_days
    test_spacing = available_days // n_splits
    
    for i in range(n_splits):
        # Calculate test end date (working backwards from end_date)
        days_from_end = i * test_spacing
        test_end = end_date - timedelta(days=days_from_end)
        train_end = test_end - timedelta(days=test_size_days)
        
        # Ensure we have minimum training data
        train_start = start_date
        train_days = (train_end - train_start).days
        
        if train_days >= min_train_days:
            splits.append((train_end, test_end))
    
    # Sort chronologically (earliest first)
    splits.sort(key=lambda x: x[0])
    
    print(f"Created {len(splits)} time series CV splits:")
    for i, (train_end, test_end) in enumerate(splits):
        train_start = start_date
        print(f"Split {i+1}: Train {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}, "
              f"Test {train_end.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    return splits

def process_single_split(df: pd.DataFrame, 
                        train_end_date: pd.Timestamp, 
                        test_end_date: pd.Timestamp,
                        weather_file: str = 'weatherDF.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a single train/test split with appropriate feature engineering.
    
    Args:
        df: Raw dataframe
        train_end_date: End of training period
        test_end_date: End of test period
        weather_file: Path to weather CSV
    
    Returns:
        Tuple of (train_df, test_df) with features engineered
    """
    print(f"Processing split: Train until {train_end_date.strftime('%Y-%m-%d')}, "
          f"Test until {test_end_date.strftime('%Y-%m-%d')}")
    
    # Create train/test split
    df['Date'] = pd.to_datetime(df['Date'])
    train_raw = df[df['Date'] < train_end_date].copy()
    test_raw = df[(df['Date'] >= train_end_date) & (df['Date'] < test_end_date)].copy()
    
    print(f"Raw split sizes - Train: {len(train_raw)}, Test: {len(test_raw)}")
    
    # ===================
    # TRAINING DATA PROCESSING (with rolling features)
    # ===================
    print("Processing training data...")
    
    # Basic preprocessing
    train_df = clean_null_load_values(train_raw)
    train_df = convert_to_utc(train_df)
    train_df = reindex_interpolate(train_df)
    train_df = add_temperature(train_df, weather_file)
    train_df = twelveHourTemp(train_df)
    train_df = hourDay(train_df)
    train_df = add_time_features(train_df)
    
    # Add rolling averages and lag features (only for training!)
    train_df = rollingAverages(train_df)
    train_df = lag_average(train_df, '1DayLag', 1)
    
    # Final processing
    train_df = numericalDate(train_df)
    
    # ===================
    # TEST DATA PROCESSING (without rolling features)
    # ===================
    print("Processing test data...")
    
    # Basic preprocessing (same as training)
    test_df = clean_null_load_values(test_raw)
    test_df = convert_to_utc(test_df)
    test_df = reindex_interpolate(test_df)
    test_df = add_temperature(test_df, weather_file)
    test_df = twelveHourTemp(test_df)
    test_df = hourDay(test_df)
    test_df = add_time_features(test_df)
    
    # Skip rolling averages and lag features to avoid data leakage
    print("Skipping rolling averages and lag features for test data (avoiding data leakage)")
    
    # Final processing
    test_df = numericalDate(test_df)
    
    print(f"Processed split sizes - Train: {len(train_df)}, Test: {len(test_df)}")
    
    return train_df, test_df

def time_series_cross_validation(raw_csv_path: str, 
                               weather_csv_path: str = 'weatherDF.csv',
                               n_splits: int = 6,
                               test_size_days: int = 7,
                               save_splits: bool = False) -> List[Dict]:
    """
    Perform time series cross-validation with growing window approach.
    
    Args:
        raw_csv_path: Path to raw data CSV
        weather_csv_path: Path to weather data CSV
        n_splits: Number of CV splits
        test_size_days: Size of test period in days
        save_splits: Whether to save individual split files
    
    Returns:
        List of dictionaries containing split info and processed DataFrames
    """
    print("=== STARTING TIME SERIES CROSS-VALIDATION ===")
    
    # Load raw data
    print(f"Loading raw data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    # Basic timestamp processing for splitting
    print("Initial timestamp processing...")
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], format='%m/%d/%Y %H:%M:%S')
    df = df.sort_values('Time Stamp').reset_index(drop=True)
    df['Date'] = df['Time Stamp'].dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded {len(df)} rows spanning {df['Date'].nunique()} unique days")
    
    # Create time series splits
    splits = create_time_series_splits(df, n_splits, test_size_days)
    
    # Process each split
    processed_splits = []
    
    for i, (train_end_date, test_end_date) in enumerate(splits):
        print(f"\n=== PROCESSING SPLIT {i+1}/{len(splits)} ===")
        
        # Process this split
        train_df, test_df = process_single_split(
            df, train_end_date, test_end_date, weather_csv_path
        )
        
        # Store split information
        split_info = {
            'split_number': i + 1,
            'train_end_date': train_end_date,
            'test_end_date': test_end_date,
            'train_df': train_df,
            'test_df': test_df,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
        
        # Save individual splits if requested
        if save_splits:
            train_filename = f'../data/processed/train_split_{i+1}.csv'
            test_filename = f'../data/processed/test_split_{i+1}.csv'
            train_df.to_csv(train_filename, index=False)
            test_df.to_csv(test_filename, index=False)
            print(f"Saved split {i+1} to {train_filename} and {test_filename}")
        
        processed_splits.append(split_info)
    
    print(f"\n=== COMPLETED {len(processed_splits)} SPLITS ===")
    return processed_splits

# Example usage and main function
def main():
    """
    Example of how to use the time series cross-validation pipeline.
    """
    # Configuration
    raw_csv_path = "../data/raw/nyc_load_aggregated_raw.csv"
    weather_csv_path = "../data/raw/weatherDF.csv"
    n_splits = 6  # Number of CV folds
    test_size_days = 7  # Each test period is 7 days
    
    # Run time series cross-validation
    splits = time_series_cross_validation(
        raw_csv_path=raw_csv_path,
        weather_csv_path=weather_csv_path,
        n_splits=n_splits,
        test_size_days=test_size_days,
        save_splits=True  # Always save splits to ../data/processed/ directory
    )
    
    # Print summary
    print("\n=== CROSS-VALIDATION SUMMARY ===")
    for split in splits:
        print(f"Split {split['split_number']}: "
              f"{split['train_size']} train samples, "
              f"{split['test_size']} test samples")
    
    return splits

if __name__ == "__main__":
    splits = main()