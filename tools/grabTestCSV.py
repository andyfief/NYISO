import pandas as pd

def split(df, testLength_days):
    df['Date'] = pd.to_datetime(df['Date'])
    train_end_date = df['Date'].max() - pd.Timedelta(days=testLength_days)

    test = df[(df['Date'] >= train_end_date)].copy()

    return test

def save_to_csv(df, filename):
    df.to_csv(filename, index=False) 
    print(f"Data saved to {filename}")
    return df

def main():
    df = pd.read_csv('../data/processed/df.csv')
    testLength_days = 7
    testSample = split(df, testLength_days)
    print(testSample.dtypes)
    testSample = testSample.drop(['Time Stamp', 'Load', 'Date'], axis=1)

    save_to_csv(testSample, '../app/testSample.csv')

if __name__ == "__main__":
    main()

