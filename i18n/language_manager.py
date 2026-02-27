"""
Language Manager - Manages language loading and caching
"""

import json
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class LanguageManager:
    """Manages language files and translations"""
    
    def __init__(self):
        self.locales_path = os.path.join(os.path.dirname(__file__), 'locales')
        self.supported_languages = {
            'en': 'English',
            'hi': 'हिंदी (Hindi)',
            'od': 'ଓଡ଼ିଆ (Odia)',
            'ta': 'தமிழ் (Tamil)',
            'te': 'తెలుగు (Telugu)',
            'bn': 'বাংলা (Bengali)',
            'gu': 'ગુજરાતી (Gujarati)',
            'mr': 'मराठी (Marathi)'
        }
        self.translations = {}
        self.default_language = 'en'
        self._load_all_languages()
        logger.info(f"✓ Language Manager initialized with {len(self.translations)} languages")
    
    def _load_all_languages(self):
        """Load all language files"""
        for lang_code in self.supported_languages.keys():
            self._load_language(lang_code)
    
    def _load_language(self, language_code: str):
        """Load a specific language file"""
        file_path = os.path.join(self.locales_path, f'{language_code}.json')
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.translations[language_code] = json.load(f)
                logger.info(f"✓ Loaded language: {language_code}")
            else:
                logger.warning(f"⚠️ Language file not found: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading language {language_code}: {str(e)}")
    
    def get_language(self, language_code: str) -> Optional[Dict]:
        """Get language translations"""
        language_code = language_code.lower()
        
        if language_code in self.translations:
            return self.translations[language_code]
        
        logger.warning(f"⚠️ Language {language_code} not found, using default {self.default_language}")
        return self.translations.get(self.default_language, {})
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages"""
        return self.supported_languages
    
    def translate(self, language_code: str, key: str, default: str = None) -> str:
        """Get translation for a key"""
        translations = self.get_language(language_code)
        
        # Handle nested keys (e.g., "weather_display.temperature")
        keys = key.split('.')
        value = translations
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default or key
        
        return value if value else (default or key)
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported"""
        return language_code.lower() in self.supported_languages