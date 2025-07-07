from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

model = joblib.load("./xgboost_model.pkl")

load_dotenv()
# NWS API doesn't require API key
location_lat = 40.7128  # NYC latitude
location_lon = -74.0060  # NYC longitude

class NWSWeatherAPI:
    def __init__(self, user_agent="FlaskWeatherApp/1.0 (contact@example.com)"):
        """Initialize the NWS Weather API client"""
        self.base_url = "https://api.weather.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json'
        })
    
    def get_gridpoint_info(self, latitude, longitude):
        """Get grid information for a given latitude/longitude"""
        url = f"{self.base_url}/points/{latitude},{longitude}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'properties' not in data:
                raise ValueError("Invalid response structure")
                
            properties = data['properties']
            
            return {
                'office': properties.get('gridId'),
                'gridX': properties.get('gridX'),
                'gridY': properties.get('gridY'),
                'forecast_hourly_url': properties.get('forecastHourly'),
                'city': properties.get('relativeLocation', {}).get('properties', {}).get('city'),
                'state': properties.get('relativeLocation', {}).get('properties', {}).get('state')
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching gridpoint info: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from NWS API")
    
    def get_hourly_forecast_by_url(self, forecast_url):
        """Get hourly forecast data using the direct URL"""
        try:
            response = self.session.get(forecast_url)
            response.raise_for_status()
            data = response.json()
            
            if 'properties' not in data or 'periods' not in data['properties']:
                raise ValueError("Invalid forecast response structure")
                
            return data['properties']['periods']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching hourly forecast: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from hourly forecast API")
    
    def fahrenheit_to_celsius(self, fahrenheit):
        """Convert Fahrenheit to Celsius"""
        return round((fahrenheit - 32) * 5/9, 1)
    
    def check_for_gaps(self, weather_data):
        """Check for gaps in hourly data"""
        for i in range(len(weather_data) - 1):
            current_time = datetime.fromisoformat(weather_data[i]['timestamp'].replace('Z', '+00:00'))
            next_time = datetime.fromisoformat(weather_data[i + 1]['timestamp'].replace('Z', '+00:00'))
            
            # Calculate time difference in hours
            time_diff = next_time - current_time
            hours_diff = time_diff.total_seconds() / 3600
            
            # Check if there's a gap (more than 1 hour difference)
            if hours_diff > 1.1:  # Allow small tolerance for timezone/DST issues
                return {'message': 'Gap in weather data detected.'}

        return {'message': 'No gaps in weather data detected.'}

    def get_weather_data(self, latitude, longitude):
        """Get complete weather data for the location"""
        try:
            # Get grid information
            grid_info = self.get_gridpoint_info(latitude, longitude)
            
            # Get hourly forecast
            forecast_periods = self.get_hourly_forecast_by_url(grid_info['forecast_hourly_url'])
            
            # Extract temperature data and convert to Celsius
            weather_data = []
            for period in forecast_periods:
                temp_f = period.get('temperature')
                if temp_f is not None:
                    temp_c = self.fahrenheit_to_celsius(temp_f)
                    
                    weather_data.append({
                        'timestamp': period.get('startTime'),
                        'temperature': temp_c
                    })
            
            # Check for gaps in the data
            gap_analysis = self.check_for_gaps(weather_data)
            
            return {
                'location': {
                    'city': grid_info['city'],
                    'state': grid_info['state'],
                    'latitude': latitude,
                    'longitude': longitude
                },
                'data': weather_data,
                'total_hours': len(weather_data),
                'temperature_unit': 'celsius',
                'gap_analysis': gap_analysis
            }
            
        except Exception as e:
            raise Exception(f"Error retrieving weather data: {e}")

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
    """Get 7-day hourly weather forecast from NWS API"""
    print("Received a request to /weather")
    try:
        # Initialize NWS API client
        nws_api = NWSWeatherAPI(user_agent="FlaskWeatherApp/1.0 (contact@example.com)")
        
        # Get weather data
        weather_data = nws_api.get_weather_data(location_lat, location_lon) 
        
        print(f"Retrieved {weather_data['total_hours']} hours of temperature data")
        print(f"Location: {weather_data['location']['city']}, {weather_data['location']['state']}")
        print(f"Gap Analysis: {weather_data['gap_analysis']['message']}")
        
        # Return the weather data with gap analysis
        return jsonify({
            "location": weather_data['location'],
            "hourly_forecast": weather_data['data'],
            "total_hours": weather_data['total_hours'],
            "temperature_unit": weather_data['temperature_unit'],
            "gap_analysis": weather_data['gap_analysis'],
            "retrieved_at": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in weather endpoint: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

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