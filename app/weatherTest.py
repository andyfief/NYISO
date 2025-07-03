#!/usr/bin/env python3
"""
National Weather Service API Hourly Temperature Data Script

This script fetches hourly temperature data from the NWS API for a given location.
The NWS API uses a two-step process:
1. Convert latitude/longitude to grid coordinates
2. Fetch hourly forecast data from the appropriate weather office

Usage: python nws_hourly_temp.py [latitude] [longitude]
"""

import requests
import json
import sys
from datetime import datetime, timedelta
import time

class NWSWeatherAPI:
    def __init__(self, user_agent="WeatherScript/1.0 (contact@example.com)"):
        """
        Initialize the NWS Weather API client
        
        Args:
            user_agent (str): User agent string (required by NWS API)
        """
        self.base_url = "https://api.weather.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json'
        })
    
    def get_gridpoint_info(self, latitude, longitude):
        """
        Get grid information for a given latitude/longitude
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            dict: Grid information including office and coordinates
        """
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
                'forecast_grid_data_url': properties.get('forecastGridData'),
                'city': properties.get('relativeLocation', {}).get('properties', {}).get('city'),
                'state': properties.get('relativeLocation', {}).get('properties', {}).get('state')
            }
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching gridpoint info: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from NWS API")
    
    def get_hourly_forecast(self, office, grid_x, grid_y):
        """
        Get hourly forecast data for a specific grid location
        
        Args:
            office (str): Weather office identifier (e.g., 'TOP')
            grid_x (int): Grid X coordinate
            grid_y (int): Grid Y coordinate
            
        Returns:
            list: List of hourly forecast periods
        """
        url = f"{self.base_url}/gridpoints/{office}/{grid_x},{grid_y}/forecast/hourly"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if 'properties' not in data or 'periods' not in data['properties']:
                raise ValueError("Invalid forecast response structure")
                
            return data['properties']['periods']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching hourly forecast: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from hourly forecast API")
    
    def get_hourly_forecast_by_url(self, forecast_url):
        """
        Get hourly forecast data using the direct URL from gridpoint info
        
        Args:
            forecast_url (str): Direct URL to hourly forecast
            
        Returns:
            list: List of hourly forecast periods
        """
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
    
    def extract_temperature_data(self, forecast_periods):
        """
        Extract only timestamp and temperature data from forecast periods
        
        Args:
            forecast_periods (list): List of forecast period dictionaries
            
        Returns:
            list: List of simplified temperature data dictionaries
        """
        temperature_data = []
        
        for period in forecast_periods:
            temp_data = {
                'datetime': period.get('startTime'),
                'temperature': period.get('temperature'),
                'temperature_unit': period.get('temperatureUnit')
            }
            temperature_data.append(temp_data)
            
        return temperature_data
    
    def get_hourly_temperature_data(self, latitude, longitude, days=7):
        """
        Get hourly temperature data for a location
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            days (int): Number of days to retrieve (default: 7, max from API)
            
        Returns:
            dict: Complete weather data including location info and hourly temperatures
        """
        try:
            # Step 1: Get grid information
            print(f"Getting grid information for {latitude}, {longitude}...")
            grid_info = self.get_gridpoint_info(latitude, longitude)
            
            print(f"Location: {grid_info['city']}, {grid_info['state']}")
            print(f"Weather Office: {grid_info['office']}")
            print(f"Grid Coordinates: {grid_info['gridX']}, {grid_info['gridY']}")
            
            # Step 2: Get all hourly forecast data (API provides up to 7 days)
            print("Fetching all available hourly forecast data...")
            forecast_periods = self.get_hourly_forecast_by_url(grid_info['forecast_hourly_url'])
            
            # Step 3: Extract temperature data (all available, up to 7 days)
            print(f"Processing {len(forecast_periods)} hourly forecast periods...")
            temperature_data = self.extract_temperature_data(forecast_periods)
            
            return {
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'city': grid_info['city'],
                    'state': grid_info['state'],
                    'weather_office': grid_info['office'],
                    'grid_coordinates': f"{grid_info['gridX']},{grid_info['gridY']}"
                },
                'forecast_data': temperature_data,
                'total_hours': len(temperature_data),
                'retrieved_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Error retrieving hourly temperature data: {e}")

def print_temperature_data(weather_data):
    """
    Print temperature data in a simplified format (timestamps and temps only)
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    location = weather_data['location']
    forecast_data = weather_data['forecast_data']
    
    print(f"\n{'='*60}")
    print(f"7-DAY HOURLY TEMPERATURE FORECAST")
    print(f"Location: {location['city']}, {location['state']}")
    print(f"Coordinates: {location['latitude']}, {location['longitude']}")
    print(f"Total Hours: {weather_data['total_hours']}")
    print(f"Retrieved: {weather_data['retrieved_at']}")
    print(f"{'='*60}")
    
    print(f"{'Date/Time':<22} {'Temperature':<12}")
    print(f"{'-'*34}")
    
    for data in forecast_data:
        # Parse datetime
        dt = datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
        time_str = dt.strftime('%m/%d/%Y %I:%M %p')
        
        temp_str = f"{data['temperature']}°{data['temperature_unit']}"
        
        print(f"{time_str:<22} {temp_str:<12}")

def print_temperature_summary(weather_data):
    """
    Print a daily summary of temperature data
    
    Args:
        weather_data (dict): Weather data dictionary
    """
    from collections import defaultdict
    
    location = weather_data['location']
    forecast_data = weather_data['forecast_data']
    
    # Group by day
    daily_temps = defaultdict(list)
    
    for data in forecast_data:
        dt = datetime.fromisoformat(data['datetime'].replace('Z', '+00:00'))
        date_key = dt.strftime('%m/%d/%Y')
        daily_temps[date_key].append(data['temperature'])
    
    print(f"\n{'='*50}")
    print(f"DAILY TEMPERATURE SUMMARY")
    print(f"Location: {location['city']}, {location['state']}")
    print(f"{'='*50}")
    
    print(f"{'Date':<12} {'High':<6} {'Low':<6} {'Hours':<6}")
    print(f"{'-'*30}")
    
    for date, temps in daily_temps.items():
        high_temp = max(temps)
        low_temp = min(temps)
        hour_count = len(temps)
        
        print(f"{date:<12} {high_temp:<6}°F {low_temp:<6}°F {hour_count:<6}")


def save_to_json(weather_data, filename=None):
    """
    Save weather data to JSON file
    
    Args:
        weather_data (dict): Weather data dictionary
        filename (str): Output filename (optional)
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"nws_hourly_forecast_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(weather_data, f, indent=2)
        print(f"\nData saved to: {filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")

def main():
    """
    Main function to run the script
    """
    # Default coordinates (you can change these or pass as arguments)
    default_lat = 40.7128  # NYC
    default_lon = -74.0060
    
    # Parse command line arguments
    if len(sys.argv) >= 3:
        try:
            latitude = float(sys.argv[1])
            longitude = float(sys.argv[2])
        except ValueError:
            print("Error: Invalid latitude/longitude values")
            sys.exit(1)
    else:
        latitude = default_lat
        longitude = default_lon
        print(f"Using default coordinates: {latitude}, {longitude}")
        print("Usage: python nws_hourly_temp.py [latitude] [longitude] [days(1-7)]")
    
    # Number of days to retrieve (API provides up to 7 days)
    days_to_retrieve = 7
    if len(sys.argv) >= 4:
        try:
            days_to_retrieve = min(int(sys.argv[3]), 7)  # Max 7 days from API
        except ValueError:
            print("Error: Invalid days value, using default (7)")
    
    try:
        # Initialize API client
        api = NWSWeatherAPI(user_agent="HourlyTempScript/1.0 (your-email@example.com)")
        
        # Get hourly temperature data for all 7 days
        weather_data = api.get_hourly_temperature_data(latitude, longitude, days_to_retrieve)
        
        # Print daily summary first
        print_temperature_summary(weather_data)
        
        # Ask user if they want to see detailed hourly data
        print(f"\nFound {weather_data['total_hours']} hours of data.")
        show_detail = input("Show detailed hourly data? (y/N): ").lower().strip()
        
        if show_detail == 'y' or show_detail == 'yes':
            print_temperature_data(weather_data)
        
        # Save to JSON file
        save_to_json(weather_data)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()