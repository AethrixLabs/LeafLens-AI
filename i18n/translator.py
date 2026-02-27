"""
Translator - Handles translation of AI outputs
"""

from typing import Dict, Any, Optional
from i18n.language_manager import LanguageManager
import logging

logger = logging.getLogger(__name__)

class Translator:
    """Translates AI outputs to regional languages"""
    
    def __init__(self):
        self.language_manager = LanguageManager()
    
    def translate_weather_response(self, weather_data: Dict, language_code: str) -> Dict:
        """Translate weather API response"""
        logger.info(f"🌐 Translating weather data to {language_code}")
        
        translated = {
            'success': weather_data.get('success'),
            'city': weather_data.get('city'),
            'timestamp': weather_data.get('timestamp'),
            'language': language_code,
            'weather': self._translate_weather_dict(weather_data.get('weather', {}), language_code),
            'risks': self._translate_risks(weather_data.get('risks', {}), language_code),
            'confidence': self._translate_confidence(weather_data.get('confidence', {}), language_code),
            'advisories': self._translate_advisories(weather_data.get('advisories', []), language_code)
        }
        
        return translated
    
    def _translate_weather_dict(self, weather: Dict, language_code: str) -> Dict:
        """Translate weather parameters"""
        translations = self.language_manager.get_language(language_code)
        weather_labels = translations.get('weather_display', {})
        
        translated = {}
        for key, value in weather.items():
            label = weather_labels.get(key, key)
            translated[label] = value
        
        return translated
    
    def _translate_risks(self, risks: Dict, language_code: str) -> Dict:
        """Translate disease risks"""
        translated = {}
        
        for disease, data in risks.items():
            translated[disease] = {
                'risk_level': data.get('risk_level'),
                'risk_score': data.get('risk_score'),
                'management_tips': data.get('management_tips'),
                'translated_level': self._translate_risk_level(data.get('risk_level'), language_code)
            }
        
        return translated
    
    def _translate_risk_level(self, level: str, language_code: str) -> str:
        """Translate risk level"""
        risk_levels = {
            'HIGH': {
                'en': 'HIGH',
                'hi': 'उच्च',
                'od': 'ଉଚ୍ଚ',
                'ta': 'அதிகம்',
                'te': 'ఎక్కువ'
            },
            'MEDIUM': {
                'en': 'MEDIUM',
                'hi': 'मध्यम',
                'od': 'ମଧ୍ୟମ',
                'ta': 'நடுத்தர',
                'te': 'మధ్యమ'
            },
            'LOW': {
                'en': 'LOW',
                'hi': 'निम्न',
                'od': 'ନିମ୍ନ',
                'ta': 'குறைந்த',
                'te': 'తక్కువ'
            }
        }
        
        return risk_levels.get(level, {}).get(language_code, level)
    
    def _translate_confidence(self, confidence: Dict, language_code: str) -> Dict:
        """Translate confidence data"""
        translations = self.language_manager.get_language(language_code)
        
        return {
            'overall_score': confidence.get('overall_score'),
            'confidence_level': confidence.get('confidence_level'),
            'recommendation': confidence.get('recommendation'),
            'label': translations.get('confidence', 'Confidence')
        }
    
    def _translate_advisories(self, advisories: list, language_code: str) -> list:
        """Translate advisory messages"""
        # For now, keep advisories as is
        # In production, you would use an API like Google Translate or custom translation
        return advisories
    
    def translate_mandi_response(self, mandi_data: Dict, language_code: str) -> Dict:
        """Translate mandi price API response"""
        logger.info(f"🌐 Translating mandi data to {language_code}")
        
        translations = self.language_manager.get_language(language_code)
        
        translated = {
            'success': mandi_data.get('success'),
            'language': language_code,
            'crop_name': mandi_data.get('crop_name'),
            'statistics': {
                'average_price_label': translations.get('avg_price', 'Average Price'),
                'average_price': mandi_data.get('statistics', {}).get('average_price'),
                'max_price_label': translations.get('max_price', 'Max Price'),
                'max_price': mandi_data.get('statistics', {}).get('max_price'),
                'min_price_label': translations.get('min_price', 'Min Price'),
                'min_price': mandi_data.get('statistics', {}).get('min_price'),
                'price_range_label': translations.get('price_range', 'Price Range'),
                'price_range': mandi_data.get('statistics', {}).get('price_range'),
                'total_records': mandi_data.get('statistics', {}).get('total_records')
            },
            'best_market': self._translate_market(mandi_data.get('best_market'), 'best', language_code),
            'worst_market': self._translate_market(mandi_data.get('worst_market'), 'worst', language_code),
            'prices': mandi_data.get('prices', [])
        }
        
        return translated
    
    def _translate_market(self, market: Dict, market_type: str, language_code: str) -> Dict:
        """Translate market information"""
        if not market:
            return None
        
        translations = self.language_manager.get_language(language_code)
        
        return {
            'title': translations.get(f'{market_type}_market', 'Market'),
            'mandi_name': market.get('mandi_name'),
            'location': market.get('location'),
            'price': market.get('price'),
            'timestamp': market.get('timestamp')
        }
    
    def get_ui_translations(self, language_code: str) -> Dict:
        """Get UI translations for frontend"""
        logger.info(f"📋 Retrieving UI translations for {language_code}")
        return self.language_manager.get_language(language_code)
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.language_manager.get_supported_languages()