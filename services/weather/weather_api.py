import requests
import json
from datetime import datetime
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherAPI:
    """Handles weather data fetching from OpenWeatherMap API"""
    
    def __init__(self, api_key: str, cache_file: str = "weather_cache.json"):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
        self.cache_file = cache_file
        self.timeout = 5
        
    def get_current_weather(self, location: str, retries: int = 3) -> Optional[Dict]:
        """Fetch current weather data for a location"""
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        for attempt in range(retries):
            try:
                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self._cache_weather(location, data)
                    logger.info(f"✓ Successfully fetched weather for {location}")
                    return data
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"✗ Failed to fetch weather after {retries} attempts")
                    return self._get_cached_weather(location)
        
        return None
    
    def _cache_weather(self, location: str, data: Dict) -> None:
        """Store weather data in local cache"""
        try:
            cache = {}
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
            except FileNotFoundError:
                pass
            
            cache[location] = {
                'data': data,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Cache write failed: {str(e)}")
    
    def _get_cached_weather(self, location: str) -> Optional[Dict]:
        """Retrieve weather data from local cache"""
        try:
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                if location in cache:
                    logger.info(f"✓ Using cached weather for {location}")
                    return cache[location]['data']
        except FileNotFoundError:
            logger.warning("Cache file not found")
        
        return None
    
    def extract_weather_params(self, weather_data: Dict) -> Dict:
        """Extract relevant weather parameters"""
        if not weather_data:
            return {}
        
        return {
            'temperature': weather_data.get('main', {}).get('temp'),
            'humidity': weather_data.get('main', {}).get('humidity'),
            'rainfall': weather_data.get('rain', {}).get('1h', 0),
            'wind_speed': weather_data.get('wind', {}).get('speed'),
            'clouds': weather_data.get('clouds', {}).get('all'),
            'condition': weather_data.get('weather', [{}])[0].get('main'),
            'pressure': weather_data.get('main', {}).get('pressure')
        }