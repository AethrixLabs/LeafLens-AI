"""
Configuration file for Weather Alert System
"""

# OpenWeatherMap API Configuration
import os
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHERMAP_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

# Cache Configuration
CACHE_FILE = "weather_cache.json"
CACHE_EXPIRY_MINUTES = 30

# Confidence Thresholds
MIN_CONFIDENCE_THRESHOLD = 0.40
MIN_ADVISORY_SEVERITY = "Medium"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "weather_alerts.log"

# Default Locations for Testing
DEFAULT_LOCATIONS = [
    "Delhi",
    "Pune",
    "Bangalore",
    "Chennai",
    "Ludhiana"
]

# Request Timeout (seconds)
REQUEST_TIMEOUT = 5

# History Window (days)
HISTORY_WINDOW_DAYS = 30