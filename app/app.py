from flask import Flask, request, jsonify
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
model = joblib.load("../models/xgboost_model.pkl")

load_dotenv()
weatherAPIKey = os.getenv("TOMORROW_API_KEY")
location="40.7128,-74.0060"

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        # Parse JSON request
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame(data)  # expecting a list of feature dicts

        # Run prediction
        prediction = model.predict(input_df)

        # Return as JSON
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/weather", methods=["GET"])
def weather_api():
    print("Received a request to /weather")
    try:
        # 1. Get hourly forecast for next 5 days
        hourly_url = f"https://api.tomorrow.io/v4/weather/forecast?location={location}&fields=temperature&units=metric&timesteps=1h&apikey={weatherAPIKey}"
        hourly_response = requests.get(hourly_url)
        hourly_data = hourly_response.json()
        print(hourly_data)

        # 2. Get daily forecast
        daily_url = f"https://api.tomorrow.io/v4/weather/forecast?location={location}&fields=temperature&units=metric&timesteps=1d&apikey={weatherAPIKey}"
        daily_response = requests.get(daily_url)
        daily_data = daily_response.json()

        # Extract just days 6 and 7
        days_6_and_7 = daily_data["timelines"]["daily"][5:7]

        print(daily_data, days_6_and_7)

        return jsonify({
            "hourly": hourly_data["timelines"]["hourly"],
            "days_6_and_7": days_6_and_7
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
if __name__ == "__main__":
    app.run(debug=True)
    

"""
currently accepting as input,
hour             int64
dayofweek        int64
month            int64
year             int64
dayofyear        int64
dayofmonth       int64
weekofyear       int64
seasonNum        int64
Temperature    float64
averageTemp    float64
"""