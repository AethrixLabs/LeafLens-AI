"""
Mandi (Market) Database - Stores crop prices across different markets
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class MandiDatabase:
    """Manages mandi price data"""
    
    def __init__(self):
        self.db_file = "data/mandi_prices.json"
        self.ensure_db_exists()
    
    def ensure_db_exists(self):
        """Create database file if it doesn't exist"""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.db_file):
            initial_data = {
                'mandis': self._get_default_mandis(),
                'crops': self._get_default_crops(),
                'prices': []
            }
            with open(self.db_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def _get_default_mandis(self) -> List[Dict]:
        """Get list of major Indian mandis including Odisha"""
        return [
            # Major Mandis
            {'id': 'delhi', 'name': 'Delhi Mandi', 'location': 'Delhi', 'state': 'Delhi'},
            {'id': 'pune', 'name': 'Pune Mandi', 'location': 'Pune', 'state': 'Maharashtra'},
            {'id': 'bangalore', 'name': 'Bangalore Mandi', 'location': 'Bangalore', 'state': 'Karnataka'},
            {'id': 'mumbai', 'name': 'Vada Mandi', 'location': 'Mumbai', 'state': 'Maharashtra'},
            {'id': 'hyderabad', 'name': 'Hyderabad Mandi', 'location': 'Hyderabad', 'state': 'Telangana'},
            {'id': 'kolkata', 'name': 'Kolkata Mandi', 'location': 'Kolkata', 'state': 'West Bengal'},
            {'id': 'jaipur', 'name': 'Jaipur Mandi', 'location': 'Jaipur', 'state': 'Rajasthan'},
            {'id': 'indore', 'name': 'Indore Mandi', 'location': 'Indore', 'state': 'Madhya Pradesh'},
            {'id': 'srinagar', 'name': 'Srinagar Mandi', 'location': 'Srinagar', 'state': 'Jammu & Kashmir'},
            {'id': 'ludhiana', 'name': 'Ludhiana Mandi', 'location': 'Ludhiana', 'state': 'Punjab'},
            
            # Odisha Mandis
            {'id': 'bhubaneswar', 'name': 'Bhubaneswar Mandi', 'location': 'Bhubaneswar', 'state': 'Odisha'},
            {'id': 'cuttack', 'name': 'Cuttack Mandi', 'location': 'Cuttack', 'state': 'Odisha'},
            {'id': 'sambalpur', 'name': 'Sambalpur Mandi', 'location': 'Sambalpur', 'state': 'Odisha'},
            {'id': 'rourkela', 'name': 'Rourkela Mandi', 'location': 'Rourkela', 'state': 'Odisha'},
            {'id': 'berhampur', 'name': 'Berhampur Mandi', 'location': 'Berhampur', 'state': 'Odisha'},
            {'id': 'balasore', 'name': 'Balasore Mandi', 'location': 'Balasore', 'state': 'Odisha'},
            {'id': 'bhadrak', 'name': 'Bhadrak Mandi', 'location': 'Bhadrak', 'state': 'Odisha'},
            {'id': 'jeypore', 'name': 'Jeypore Mandi', 'location': 'Jeypore', 'state': 'Odisha'},
            {'id': 'baripada', 'name': 'Baripada Mandi', 'location': 'Baripada', 'state': 'Odisha'},
            {'id': 'bhawanipatna', 'name': 'Bhawanipatna Mandi', 'location': 'Bhawanipatna', 'state': 'Odisha'},
            {'id': 'angul', 'name': 'Angul Mandi', 'location': 'Angul', 'state': 'Odisha'},
            {'id': 'dhenkanal', 'name': 'Dhenkanal Mandi', 'location': 'Dhenkanal', 'state': 'Odisha'},
            {'id': 'bargarh', 'name': 'Bargarh Mandi', 'location': 'Bargarh', 'state': 'Odisha'},
            {'id': 'jharsuguda', 'name': 'Jharsuguda Mandi', 'location': 'Jharsuguda', 'state': 'Odisha'},
            {'id': 'koraput', 'name': 'Koraput Mandi', 'location': 'Koraput', 'state': 'Odisha'}
        ]
    
    def _get_default_crops(self) -> List[Dict]:
        """Get list of common crops"""
        return [
            {'id': 'wheat', 'name': 'Wheat', 'unit': 'kg'},
            {'id': 'rice', 'name': 'Rice', 'unit': 'kg'},
            {'id': 'potato', 'name': 'Potato', 'unit': 'kg'},
            {'id': 'onion', 'name': 'Onion', 'unit': 'kg'},
            {'id': 'tomato', 'name': 'Tomato', 'unit': 'kg'},
            {'id': 'corn', 'name': 'Corn', 'unit': 'kg'},
            {'id': 'cotton', 'name': 'Cotton', 'unit': 'kg'},
            {'id': 'sugarcane', 'name': 'Sugarcane', 'unit': 'quintal'},
            {'id': 'soybean', 'name': 'Soybean', 'unit': 'kg'},
            {'id': 'chickpea', 'name': 'Chickpea', 'unit': 'kg'},
            {'id': 'lentils', 'name': 'Lentils', 'unit': 'kg'},
            {'id': 'pepper', 'name': 'Pepper', 'unit': 'kg'},
            {'id': 'garlic', 'name': 'Garlic', 'unit': 'kg'},
            {'id': 'ginger', 'name': 'Ginger', 'unit': 'kg'},
            {'id': 'turmeric', 'name': 'Turmeric', 'unit': 'kg'},
        ]
    
    def add_price(self, crop_id: str, mandi_id: str, price: float, quantity: float = 1.0) -> Dict:
        """Add a new price entry"""
        data = self._read_db()
        
        price_entry = {
            'id': len(data['prices']) + 1,
            'crop_id': crop_id,
            'mandi_id': mandi_id,
            'price': price,
            'quantity': quantity,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        data['prices'].append(price_entry)
        self._write_db(data)
        
        return price_entry
    
    def get_prices(self, crop_id: Optional[str] = None, mandi_id: Optional[str] = None) -> List[Dict]:
        """Get prices filtered by crop and/or mandi"""
        data = self._read_db()
        prices = data['prices']
        
        if crop_id:
            prices = [p for p in prices if p['crop_id'] == crop_id]
        
        if mandi_id:
            prices = [p for p in prices if p['mandi_id'] == mandi_id]
        
        # Sort by timestamp descending (newest first)
        prices.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return prices
    
    def get_latest_prices(self, crop_id: str) -> Dict:
        """Get latest prices for a crop across all mandis"""
        data = self._read_db()
        crop_prices = [p for p in data['prices'] if p['crop_id'] == crop_id]
        
        # Get latest price for each mandi
        latest = {}
        for price in crop_prices:
            mandi_id = price['mandi_id']
            if mandi_id not in latest or price['timestamp'] > latest[mandi_id]['timestamp']:
                latest[mandi_id] = price
        
        return latest
    
    def get_price_stats(self, crop_id: str) -> Dict:
        """Get price statistics for a crop"""
        prices = self.get_prices(crop_id)
        
        if not prices:
            return {
                'crop_id': crop_id,
                'average': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
        
        price_values = [p['price'] for p in prices]
        
        return {
            'crop_id': crop_id,
            'average': sum(price_values) / len(price_values),
            'min': min(price_values),
            'max': max(price_values),
            'count': len(price_values),
            'latest': prices[0] if prices else None
        }
    
    def get_mandis(self) -> List[Dict]:
        """Get all mandis"""
        data = self._read_db()
        return data['mandis']
    
    def get_crops(self) -> List[Dict]:
        """Get all crops"""
        data = self._read_db()
        return data['crops']
    
    def _read_db(self) -> Dict:
        """Read database file"""
        with open(self.db_file, 'r') as f:
            return json.load(f)
    
    def _write_db(self, data: Dict):
        """Write database file"""
        with open(self.db_file, 'w') as f:
            json.dump(data, f, indent=2)