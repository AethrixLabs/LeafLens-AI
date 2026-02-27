"""
Weather Service – 3-Day Forecast Integration for IoT Dashboard.

Provides weather forecast data for selected locations using OpenWeatherMap API.
- Fetches 5-day forecast data
- Extracts next 3 calendar days (excluding today)
- Computes daily aggregates: temperature, humidity, rainfall
- Handles API errors with proper logging and exceptions
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from statistics import mean

import requests

logger = logging.getLogger("leaflens")


class WeatherServiceError(Exception):
    """Custom exception for weather service failures."""
    pass


class LocationNotFoundError(WeatherServiceError):
    """Exception raised when location cannot be found."""
    pass


class APIKeyMissingError(WeatherServiceError):
    """Exception raised when API key is not configured."""
    pass


def get_3_day_forecast(city: str) -> List[Dict[str, float]]:
    """
    Fetch 3-day weather forecast for a given city.

    Retrieves OpenWeatherMap 5-day forecast, extracts next 3 calendar days
    (excluding today), and computes daily aggregates for:
    - Average temperature (°C, rounded to 2 decimals)
    - Average humidity (%, rounded to 2 decimals)
    - Total rainfall (mm, rounded to 2 decimals)

    Args:
        city: City name (e.g., "Bhubaneswar", "Delhi")

    Returns:
        List of dictionaries with daily forecast:
        [
            {
                "date": "YYYY-MM-DD",
                "avg_temp": float,
                "avg_humidity": float,
                "total_rain_mm": float
            },
            ...
        ]

    Raises:
        APIKeyMissingError: If OPENWEATHER_API_KEY environment variable is not set
        LocationNotFoundError: If city cannot be found on OpenWeatherMap
        WeatherServiceError: If API request fails or data processing fails

    Examples:
        >>> forecast = get_3_day_forecast("Bhubaneswar")
        >>> len(forecast)
        3
        >>> forecast[0]["date"]
        "2026-02-25"
    """
    # Validate API key
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        logger.error(
            "OPENWEATHER_API_KEY environment variable not set. "
            "Please configure the API key before using weather forecasts."
        )
        raise APIKeyMissingError(
            "OPENWEATHER_API_KEY environment variable is not configured"
        )

    # OpenWeatherMap 5-day forecast endpoint
    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # Temperature in Celsius, rain in mm
    }

    try:
        logger.info(f"Fetching 3-day forecast for city: {city}")
        
        response = requests.get(
            forecast_url,
            params=params,
            timeout=10
        )

        # Handle HTTP errors
        if response.status_code == 404:
            logger.warning(f"City not found: {city}")
            raise LocationNotFoundError(f"City '{city}' not found on OpenWeatherMap")
        
        if response.status_code == 401:
            logger.error("Invalid API key provided")
            raise APIKeyMissingError("Invalid OPENWEATHER_API_KEY")
        
        if response.status_code != 200:
            logger.error(
                f"OpenWeatherMap API error: {response.status_code} - {response.text}"
            )
            raise WeatherServiceError(
                f"OpenWeatherMap API returned status {response.status_code}"
            )

        data = response.json()
        
        # Extract forecast list
        forecast_list = data.get("list", [])
        if not forecast_list:
            raise WeatherServiceError("No forecast data available for this location")

        logger.info(f"Successfully fetched forecast data with {len(forecast_list)} entries")

        # Parse forecast data and group by date
        forecast_by_date = _group_forecast_by_date(forecast_list)
        
        # Get next 3 calendar days (skip today)
        three_day_forecast = _extract_next_three_days(forecast_by_date)
        
        logger.info(
            f"Extracted 3-day forecast for {city}: "
            f"{len(three_day_forecast)} days available"
        )
        
        return three_day_forecast

    except requests.exceptions.Timeout:
        error_msg = f"Timeout fetching forecast for {city} (10s timeout)"
        logger.error(error_msg)
        raise WeatherServiceError(error_msg)
    
    except requests.exceptions.ConnectionError:
        error_msg = f"Connection error fetching forecast for {city}"
        logger.error(error_msg)
        raise WeatherServiceError(error_msg)
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed fetching forecast for {city}: {str(e)}"
        logger.error(error_msg)
        raise WeatherServiceError(error_msg)
    
    except (LocationNotFoundError, APIKeyMissingError):
        # Re-raise custom exceptions
        raise
    
    except Exception as e:
        error_msg = f"Unexpected error fetching forecast for {city}: {str(e)}"
        logger.exception(error_msg)
        raise WeatherServiceError(error_msg)


def _group_forecast_by_date(forecast_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group forecast entries by calendar date.

    OpenWeatherMap returns data in 3-hour intervals. This function groups
    all entries for the same calendar day.

    Args:
        forecast_list: List of forecast entries from OpenWeatherMap API

    Returns:
        Dictionary mapping date string (YYYY-MM-DD) to list of forecast entries
    """
    grouped = {}
    
    for entry in forecast_list:
        try:
            # Parse timestamp (format: "2026-02-25 12:00:00")
            dt_str = entry.get("dt_txt")
            if not dt_str:
                continue
            
            # Extract date part (YYYY-MM-DD)
            date = dt_str.split(" ")[0]
            
            if date not in grouped:
                grouped[date] = []
            
            grouped[date].append(entry)
        
        except (KeyError, ValueError, IndexError):
            logger.warning(f"Skipping malformed forecast entry: {entry}")
            continue
    
    return grouped


def _extract_next_three_days(
    forecast_by_date: Dict[str, List[Dict]]
) -> List[Dict[str, float]]:
    """
    Extract next 3 calendar days from grouped forecast data.

    Skips today and returns next 3 available days with computed aggregates.

    Args:
        forecast_by_date: Dictionary mapping dates to forecast entries

    Returns:
        List of up to 3 days with aggregated weather data
    """
    today = datetime.utcnow().date()
    sorted_dates = sorted(forecast_by_date.keys())
    
    # Filter out today and future dates
    future_dates = [d for d in sorted_dates if d > str(today)]
    
    # Take next 3 days
    next_three_days = future_dates[:3]
    
    result = []
    
    for date_str in next_three_days:
        entries = forecast_by_date[date_str]
        
        try:
            # Compute daily aggregates
            daily_data = _compute_daily_aggregate(entries)
            daily_data["date"] = date_str
            result.append(daily_data)
        
        except Exception as e:
            logger.warning(f"Error computing aggregate for {date_str}: {str(e)}")
            continue
    
    return result


def _compute_daily_aggregate(entries: List[Dict]) -> Dict[str, float]:
    """
    Compute daily aggregates from 3-hour interval forecast entries.

    For each day, computes:
    - Average temperature (°C)
    - Average humidity (%)
    - Total rainfall (mm)

    Args:
        entries: List of 3-hour forecast entries for a single day

    Returns:
        Dictionary with avg_temp, avg_humidity, total_rain_mm (all rounded to 2 decimals)

    Raises:
        KeyError: If required fields are missing from entry
    """
    temperatures = []
    humidities = []
    total_rain = 0.0
    
    for entry in entries:
        # Extract main weather data
        main = entry.get("main", {})
        
        # Temperature in Celsius
        temp = main.get("temp")
        if temp is not None:
            temperatures.append(temp)
        
        # Humidity percentage
        humidity = main.get("humidity")
        if humidity is not None:
            humidities.append(humidity)
        
        # Rainfall: check both "rain" and "snow"
        rain = entry.get("rain", {})
        if isinstance(rain, dict):
            total_rain += rain.get("3h", 0.0)  # 3-hour rainfall
        
        snow = entry.get("snow", {})
        if isinstance(snow, dict):
            # Convert snow to water equivalent (typically 1 mm snow ≈ 0.1 mm water)
            total_rain += snow.get("3h", 0.0) * 0.1
    
    # Compute averages
    avg_temp = round(mean(temperatures), 2) if temperatures else 0.0
    avg_humidity = round(mean(humidities), 2) if humidities else 0.0
    total_rain_mm = round(total_rain, 2)
    
    return {
        "avg_temp": avg_temp,
        "avg_humidity": avg_humidity,
        "total_rain_mm": total_rain_mm
    }
