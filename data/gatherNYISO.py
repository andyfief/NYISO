import pandas as pd
import requests
import zipfile
import io
import os
from datetime import datetime, timedelta

# Replace this with your CSV URL from NYISO
url = 'https://mis.nyiso.com/public/csv/palIntegrated/20250401palIntegrated_csv.zip'

def process_nyiso_csv(url, zone='N.Y.C.', output_file='nyc_load_aggregated.csv'):
    # Step 1: Download ZIP file from NYISO
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to download:", url)
        return
    
    # Step 2: Unzip the file
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall("nyiso_temp")  # Extracts to a temporary folder

    # Step 3: Process each CSV file in the ZIP
    for csv_file in zip_ref.namelist():
        csv_path = os.path.join("nyiso_temp", csv_file)
        print(f"Processing file: {csv_path}")

        # Step 4: Load the CSV file into pandas
        df = pd.read_csv(csv_path)

        # Step 5: Clean and filter for zone (e.g., 'N.Y.C.')
        justNYC = df[df['Name'] == zone].copy()

        # Step 6: Append the filtered data to the output file
        if os.path.exists(output_file):
            justNYC.to_csv(output_file, mode='a', header=False, index=False)
        else:
            justNYC.to_csv(output_file, index=False)

        print(f"Processed and appended data for: {zone} from {csv_file}")

    # Step 7: Clean up the temporary folder
    for file in os.listdir("nyiso_temp"):
        os.remove(os.path.join("nyiso_temp", file))
    print(f"Finished processing all files. Data saved in {output_file}.")


def get_nyiso_urls(start_date='20010601', end_date = '20250401'):
    # Step 1: Parse start and end date
    start_year = int(start_date[:4])
    start_month = int(start_date[4:6])
    end_year = int(end_date[:4])
    end_month = int(end_date[4:6])
    
    # Step 2: Generate list of URLs for each month
    urls = []
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    while current_date <= end_date:
        # Format the date as 'yyyyMMdd' (first of the month)
        date_str = current_date.strftime('%Y%m01')
        url = f'https://mis.nyiso.com/public/csv/palIntegrated/{date_str}palIntegrated_csv.zip'
        urls.append(url)
        
        # Move to the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    return urls

# Loop through all URLs and process the files
nyiso_urls = get_nyiso_urls()  # Get the list of URLs from 2001-06-01 to the current month
print(nyiso_urls)

for url in nyiso_urls:
    process_nyiso_csv(url)
