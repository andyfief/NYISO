import csv
import pandas as pd


def checkMissing(df):
    df = df.reset_index()
    df['time'] = pd.to_datetime(df['time'])

    # Create expected full hourly range
    full_range = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq='H')

    # Find missing timestamps
    missing = full_range.difference(df['time'])

    print(f"\nTotal expected hours: {len(full_range)}")
    print(f"Total available rows: {len(df)}")
    print(f"Missing hours: {len(missing)}")

    if not missing.empty:
        print("Missing timestamps:")
        print(missing[:10])  # Show first 10
    else:
        print("✅ All hourly records are present!")


def main():
    csv_path = "weatherDF.csv"
    print("Loading weather csv file...")
    df = pd.read_csv(csv_path)

    checkMissing(df)




if __name__ == "__main__":
    main()