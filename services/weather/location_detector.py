"""
Location Detector - Detects user's current location using IP geolocation
"""

import requests
import logging
from typing import Dict, Optional
import random

logger = logging.getLogger(__name__)

class LocationDetector:
    """Detects user location using multiple geolocation APIs"""
    
    def __init__(self):
        # Multiple geolocation API endpoints to try
        self.apis = [
            {
                'name': 'ip-api.com',
                'url': 'http://ip-api.com/json/',
                'parser': self._parse_ip_api
            },
            {
                'name': 'ipapi.co',
                'url': 'https://ipapi.co/json/',
                'parser': self._parse_ipapi_co
            },
            {
                'name': 'geoip-db.com',
                'url': 'https://geoip-db.com/json/',
                'parser': self._parse_geoip_db
            },
            {
                'name': 'ipinfo.io',
                'url': 'https://ipinfo.io/json',
                'parser': self._parse_ipinfo
            }
        ]
        
        # Default cities if location detection fails
        self.default_cities = [
            {'city': 'Delhi', 'country': 'India', 'region': 'Delhi', 'latitude': 28.7041, 'longitude': 77.1025},
            {'city': 'Mumbai', 'country': 'India', 'region': 'Maharashtra', 'latitude': 19.0760, 'longitude': 72.8777},
            {'city': 'Bangalore', 'country': 'India', 'region': 'Karnataka', 'latitude': 12.9716, 'longitude': 77.5946},
            {'city': 'Bhubaneswar', 'country': 'India', 'region': 'Odisha', 'latitude': 20.2961, 'longitude': 85.8245},
            {'city': 'Kolkata', 'country': 'India', 'region': 'West Bengal', 'latitude': 22.5726, 'longitude': 88.3639},
            {'city': 'Hyderabad', 'country': 'India', 'region': 'Telangana', 'latitude': 17.3850, 'longitude': 78.4867},
            {'city': 'Chennai', 'country': 'India', 'region': 'Tamil Nadu', 'latitude': 13.0827, 'longitude': 80.2707},
            {'city': 'Pune', 'country': 'India', 'region': 'Maharashtra', 'latitude': 18.5204, 'longitude': 73.8567}
        ]
        
        logger.info("✓ Location Detector initialized with 4 geolocation APIs")
    
    def get_current_location(self) -> Optional[Dict]:
        """
        Detect current location using IP geolocation
        Returns location data or None if detection fails
        """
        logger.info("🌍 Attempting to detect location...")
        
        # Try each API in sequence
        for api in self.apis:
            try:
                logger.info(f"🔄 Trying {api['name']}...")
                location = self._fetch_from_api(api)
                
                if location:
                    logger.info(f"✓ Location detected via {api['name']}: {location['city']}, {location['country']}")
                    return location
            except Exception as e:
                logger.warning(f"⚠️ {api['name']} failed: {str(e)}")
                continue
        
        logger.warning("⚠️ All location detection methods failed")
        return None
    
    def _fetch_from_api(self, api: Dict) -> Optional[Dict]:
        """Fetch location from a specific API"""
        try:
            logger.debug(f"Requesting from {api['url']}")
            
            response = requests.get(
                api['url'],
                timeout=5,
                headers={
                    'User-Agent': 'LeafLens-AI/1.0 (Weather Alert System)'
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Response from {api['name']}: {data}")
                
                # Parse response based on API
                location = api['parser'](data)
                
                if location:
                    return location
        except requests.exceptions.Timeout:
            logger.warning(f"⏱️ {api['name']} timeout (5s)")
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 {api['name']} connection error")
        except Exception as e:
            logger.warning(f"❌ Error fetching from {api['name']}: {str(e)}")
        
        return None
    
    def _parse_ip_api(self, data: Dict) -> Optional[Dict]:
        """Parse response from ip-api.com"""
        try:
            if data.get('status') == 'fail':
                logger.debug(f"ip-api failed: {data.get('message')}")
                return None
            
            city = data.get('city', 'Unknown')
            if not city or city == '':
                return None
            
            return {
                'city': city,
                'country': data.get('country', 'Unknown'),
                'region': data.get('regionName', 'Unknown'),
                'latitude': float(data.get('lat', 0)),
                'longitude': float(data.get('lon', 0)),
                'isp': data.get('isp', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error parsing ip-api response: {e}")
            return None
    
    def _parse_ipapi_co(self, data: Dict) -> Optional[Dict]:
        """Parse response from ipapi.co"""
        try:
            if data.get('error'):
                logger.debug(f"ipapi.co error: {data.get('error_message')}")
                return None
            
            city = data.get('city', 'Unknown')
            if not city or city == '':
                return None
            
            return {
                'city': city,
                'country': data.get('country_name', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'latitude': float(data.get('latitude', 0)),
                'longitude': float(data.get('longitude', 0)),
                'isp': data.get('org', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error parsing ipapi.co response: {e}")
            return None
    
    def _parse_geoip_db(self, data: Dict) -> Optional[Dict]:
        """Parse response from geoip-db.com"""
        try:
            city = data.get('city', 'Unknown')
            if not city or city == '':
                return None
            
            return {
                'city': city,
                'country': data.get('country_name', 'Unknown'),
                'region': data.get('state', 'Unknown'),
                'latitude': float(data.get('latitude', 0)),
                'longitude': float(data.get('longitude', 0)),
                'isp': 'Unknown'
            }
        except Exception as e:
            logger.error(f"Error parsing geoip-db response: {e}")
            return None
    
    def _parse_ipinfo(self, data: Dict) -> Optional[Dict]:
        """Parse response from ipinfo.io"""
        try:
            if data.get('error'):
                logger.debug(f"ipinfo error: {data.get('error')}")
                return None
            
            city = data.get('city', 'Unknown')
            if not city or city == '':
                return None
            
            loc = data.get('loc', '0,0').split(',')
            
            return {
                'city': city,
                'country': data.get('country', 'Unknown'),
                'region': data.get('region', 'Unknown'),
                'latitude': float(loc[0]) if len(loc) > 0 else 0,
                'longitude': float(loc[1]) if len(loc) > 1 else 0,
                'isp': data.get('org', 'Unknown')
            }
        except Exception as e:
            logger.error(f"Error parsing ipinfo response: {e}")
            return None
    
    def get_location_by_city(self, city: str) -> Optional[Dict]:
        """Get coordinates for a city name"""
        logger.info(f"📍 Getting coordinates for: {city}")
        
        try:
            response = requests.get(
                'https://api.openweathermap.org/geo/1.0/direct',
                params={
                    'q': city,
                    'limit': 1
                },
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    city_data = data[0]
                    return {
                        'city': city_data.get('name', city),
                        'country': city_data.get('country', 'Unknown'),
                        'region': city_data.get('state', 'Unknown'),
                        'latitude': city_data.get('lat', 0),
                        'longitude': city_data.get('lon', 0)
                    }
        except Exception as e:
            logger.error(f"Error getting city coordinates: {e}")
        
        return None
    
    def get_fallback_location(self) -> Dict:
        """Get a random fallback location"""
        location = random.choice(self.default_cities)
        logger.info(f"📌 Using fallback location: {location['city']}")
        return location
    
    def detect_location_with_fallback(self) -> Dict:
        """Detect location with fallback to default city"""
        location = self.get_current_location()
        
        if location:
            return location
        else:
            logger.warning("⚠️ Location detection failed, using fallback")
            return self.get_fallback_location()
    
    def get_default_cities(self) -> list:
        """Get list of default cities"""
        return self.default_cities