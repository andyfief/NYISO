# NYC Power Grid Load Forecasting

A machine learning web app that predicts NYC's power grid load for the next 7 days using real-time weather data and historical patterns. Built with XGBoost and the National Weather Service API.
Sliding-window cross validation was used to compute an accuracy of 97.1%, using the Mean Absolute Percentage Error Error across 52 models: 1 per week for a year.

## How It Works

1. **Historical Data**: Scrapes 20+ years of NYC power grid data from NYISO
2. **Weather Integration**: Merges histoical temperature data with power consumption patterns
3. **Feature Engineering**: Creates time-based features (hour, day, season) and temperature averages
4. **Model Training**: XGBoost learns patterns and makes predictions that are evaluated with a sliding window
5. **Real-time Forecasting**: NWS API provides current weather → model predicts next 7 days
6. **Web Interface**: Clean charts show both weather forecast and power predictions

## Quick Start

### Local Setup
```bash
# Clone and install dependencies
pip install flask pandas xgboost scikit-learn joblib matplotlib numpy python-dotenv flask-cors requests dayjs

# Run the Flask app
python app.py

# Open your browser to http://localhost:5000
```

### Docker
```bash
# Build the image
docker build -t nyiso .

# Run the container
docker run -p 5000:5000 nyiso

# Open your browser to http://localhost:5000
```

## APIs Used

- **National Weather Service (NWS) API** - Free, no API key needed. Pulls 7-day hourly weather forecasts for NYC
- **NYISO API** - Historical power grid load data from New York Independent System Operator

## The Model

**XGBoost Regressor** with these parameters:
- 1000 estimators
- Learning rate: 0.01
- Max depth: 10
- Early stopping: 50 rounds
- Objective: Squared error regression

## Training Data Columns

The model trains on these features:
- `hour` - Hour of day (0-23)
- `dayofweek` - Day of week (0-6)
- `month` - Month (1-12)
- `year` - Year
- `dayofyear` - Day of year (1-365/366)
- `dayofmonth` - Day of month (1-31)
- `weekofyear` - Week of year (1-52)
- `seasonNum` - Season number (1-4)
- `Temperature` - Current temperature (°C)
- `averageTemp` - Daily average temperature 9AM-9PM (°C)

Target variable: `Load` (power demand in MWh)

## Tools Created

1. **Data Collection Pipeline** (`gatherNYISO.py`) - Downloads and processes historical NYISO data
2. **Weather Data Collector** (`weather.py`) - Fetches historical weather using Meteostat
3. **Data Processor** (`buildCSV.py`) - Cleans, merges, and engineers features from raw data
4. **Model Trainer** (`XG.py`) - Trains XGBoost model with sliding window validation
5. **Web App** (`app.py`) - Flask API serving predictions and weather data
6. **Frontend** (`frontend.js`) - Chart.js-powered dashboard

## Libraries Used (Python)

**Core ML & Data:**
- `pandas` - Data manipulation
- `xgboost` - Gradient boosting model
- `scikit-learn` - ML utilities and metrics
- `numpy` - Numerical computing
- `joblib` - Model serialization

**Web & API:**
- `flask` - Web framework
- `flask-cors` - Cross-origin requests
- `requests` - HTTP requests
- `python-dotenv` - Environment variables

**Time & Weather:**
- `meteostat` - Historical weather data
- `pytz` - Timezone handling
- `holidays` - Holiday detection

**Visualization:**
- `matplotlib` - Plotting and analysis

## Project Structure

```
├── /app
  ├── app.py               # Flask web server
  ├── /static
      ├── frontend.js      # Client-side dashboard
  ├── /templates
      ├── index.html
├── /models       
    ├── XG.py              #XGBoost training evalutating, and saving
    ├── xgboost_model.pkl  # Trained model (generated)
├── /tools
    ├── buildCSV.py        # Data preprocessing
    ├── gatherNYISO.py     # NYISO data collection
    ├── weather.py         # Weather data collection
    ├── cleanWeather.py    # Weather data validation
```

## Notes

- **Time Range**: 2001-2025
- **Resolution**: Hourly predictions
- **Geography**: NYC zone only
- **Validation**: Sliding window testing across multiple time periods
- **Performance**: RMSE varies by season, typically 50-200 MWh
- **Evaluation and Usage**: Accuracy scores mentioned are calculated with historical weather data. For forecasting, model accuracy should be expected to worsen with inaccurate weather forecasts.
