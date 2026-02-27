"""
Mandi Price API - Handles market price operations
"""

import logging
from typing import List, Dict, Optional
from services.mandi.mandi_db import MandiDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MandiPriceAPI:
    """API for mandi price operations"""
    
    def __init__(self):
        self.db = MandiDatabase()
        if not self.db.get_prices():
            self.load_sample_data()
        logger.info("✓ Mandi Price API initialized")
    
    def add_price_entry(self, crop_id: str, mandi_id: str, price: float) -> Dict:
        """Add a new price entry to the market"""
        try:
            logger.info(f"Adding price: {crop_id} @ {mandi_id} - Rs.{price}")
            
            entry = self.db.add_price(crop_id, mandi_id, price)
            
            logger.info(f"✓ Price entry added successfully")
            return {
                'success': True,
                'data': entry,
                'message': f'Price added for {crop_id} at {mandi_id}'
            }
        except Exception as e:
            logger.error(f"Error adding price: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_market_prices(self, crop_id: str) -> Dict:
        """Get current prices for a crop across all markets"""
        try:
            logger.info(f"Fetching market prices for {crop_id}")
            
            latest_prices = self.db.get_latest_prices(crop_id)
            
            # Enrich with mandi and crop info
            mandis = {m['id']: m for m in self.db.get_mandis()}
            crops = {c['id']: c for c in self.db.get_crops()}
            
            enriched_prices = []
            for mandi_id, price_data in latest_prices.items():
                enriched_prices.append({
                    **price_data,
                    'mandi_name': mandis.get(mandi_id, {}).get('name', 'Unknown'),
                    'location': mandis.get(mandi_id, {}).get('location', 'Unknown'),
                    'crop_name': crops.get(crop_id, {}).get('name', 'Unknown'),
                    'unit': crops.get(crop_id, {}).get('unit', 'kg')
                })
            
            return {
                'success': True,
                'crop_id': crop_id,
                'prices': enriched_prices,
                'count': len(enriched_prices)
            }
        except Exception as e:
            logger.error(f"Error getting market prices: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_price_comparison(self, crop_id: str) -> Dict:
        """Get price comparison and statistics"""
        try:
            logger.info(f"Getting price comparison for {crop_id}")
            
            stats = self.db.get_price_stats(crop_id)
            latest_prices = self.db.get_latest_prices(crop_id)
            
            # Get mandis info
            mandis = {m['id']: m for m in self.db.get_mandis()}
            crops = {c['id']: c for c in self.db.get_crops()}
            
            # Find best and worst prices
            if latest_prices:
                prices_list = list(latest_prices.values())
                best_price = max(prices_list, key=lambda x: x['price'])
                worst_price = min(prices_list, key=lambda x: x['price'])
            else:
                best_price = worst_price = None
            
            return {
                'success': True,
                'crop_id': crop_id,
                'crop_name': crops.get(crop_id, {}).get('name', 'Unknown'),
                'unit': crops.get(crop_id, {}).get('unit', 'kg'),
                'statistics': {
                    'average_price': round(stats['average'], 2),
                    'min_price': stats['min'],
                    'max_price': stats['max'],
                    'price_range': stats['max'] - stats['min'],
                    'total_records': stats['count']
                },
                'best_market': {
                    'mandi_name': mandis.get(best_price['mandi_id'], {}).get('name') if best_price else None,
                    'location': mandis.get(best_price['mandi_id'], {}).get('location') if best_price else None,
                    'price': best_price['price'] if best_price else None,
                    'timestamp': best_price['timestamp'] if best_price else None
                } if best_price else None,
                'worst_market': {
                    'mandi_name': mandis.get(worst_price['mandi_id'], {}).get('name') if worst_price else None,
                    'location': mandis.get(worst_price['mandi_id'], {}).get('location') if worst_price else None,
                    'price': worst_price['price'] if worst_price else None,
                    'timestamp': worst_price['timestamp'] if worst_price else None
                } if worst_price else None
            }
        except Exception as e:
            logger.error(f"Error getting price comparison: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_mandi_prices(self, mandi_id: str) -> Dict:
        """Get all prices for a specific mandi"""
        try:
            logger.info(f"Fetching prices for mandi: {mandi_id}")
            
            prices = self.db.get_prices(mandi_id=mandi_id)
            
            # Get mandi and crop info
            mandis = {m['id']: m for m in self.db.get_mandis()}
            crops = {c['id']: c for c in self.db.get_crops()}
            
            enriched_prices = []
            for price in prices:
                enriched_prices.append({
                    **price,
                    'crop_name': crops.get(price['crop_id'], {}).get('name', 'Unknown'),
                    'unit': crops.get(price['crop_id'], {}).get('unit', 'kg'),
                    'mandi_name': mandis.get(mandi_id, {}).get('name', 'Unknown'),
                    'location': mandis.get(mandi_id, {}).get('location', 'Unknown')
                })
            
            return {
                'success': True,
                'mandi_id': mandi_id,
                'mandi_name': mandis.get(mandi_id, {}).get('name', 'Unknown'),
                'location': mandis.get(mandi_id, {}).get('location', 'Unknown'),
                'prices': enriched_prices,
                'count': len(enriched_prices)
            }
        except Exception as e:
            logger.error(f"Error getting mandi prices: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_all_mandis(self) -> List[Dict]:
        """Get all available mandis"""
        return self.db.get_mandis()
    
    def get_all_crops(self) -> List[Dict]:
        """Get all available crops"""
        return self.db.get_crops()
    
    def load_sample_data(self) -> Dict:
        """Load sample price data for demonstration"""
        try:
            logger.info("Loading sample data...")
            
            sample_prices = [
                # Wheat prices
                ('wheat', 'delhi', 2200),
                ('wheat', 'ludhiana', 2150),
                ('wheat', 'bangalore', 2350),
                ('wheat', 'pune', 2280),
                ('wheat', 'jaipur', 2120),
                ('wheat', 'indore', 2180),
                ('wheat', 'hyderabad', 2300),
                ('wheat', 'kolkata', 2350),
                ('wheat', 'mumbai', 2400),
                ('wheat', 'srinagar', 2500),
                ('wheat', 'bhubaneswar', 2220),
                ('wheat', 'cuttack', 2210),
                ('wheat', 'sambalpur', 2195),
                ('wheat', 'rourkela', 2240),
                ('wheat', 'berhampur', 2260),
                ('wheat', 'balasore', 2230),
                ('wheat', 'bhadrak', 2215),
                ('wheat', 'jeypore', 2200),
                ('wheat', 'baripada', 2225),
                ('wheat', 'bhawanipatna', 2185),
                ('wheat', 'angul', 2205),
                ('wheat', 'dhenkanal', 2190),
                ('wheat', 'bargarh', 2170),
                ('wheat', 'jharsuguda', 2200),
                ('wheat', 'koraput', 2250),
                
                # Rice prices
                ('rice', 'delhi', 3500),
                ('rice', 'ludhiana', 3400),
                ('rice', 'bangalore', 3800),
                ('rice', 'pune', 3600),
                ('rice', 'kolkata', 3900),
                ('rice', 'hyderabad', 3700),
                ('rice', 'mumbai', 3900),
                ('rice', 'srinagar', 4200),
                ('rice', 'jaipur', 3550),
                ('rice', 'indore', 3650),
                ('rice', 'bhubaneswar', 3680),
                ('rice', 'cuttack', 3650),
                ('rice', 'sambalpur', 3620),
                ('rice', 'rourkela', 3700),
                ('rice', 'berhampur', 3750),
                ('rice', 'balasore', 3700),
                ('rice', 'bhadrak', 3680),
                ('rice', 'jeypore', 3600),
                ('rice', 'baripada', 3720),
                ('rice', 'bhawanipatna', 3580),
                ('rice', 'angul', 3640),
                ('rice', 'dhenkanal', 3610),
                ('rice', 'bargarh', 3550),
                ('rice', 'jharsuguda', 3600),
                ('rice', 'koraput', 3750),
                
                # Potato prices
                ('potato', 'delhi', 1800),
                ('potato', 'ludhiana', 1700),
                ('potato', 'bangalore', 2000),
                ('potato', 'pune', 1900),
                ('potato', 'jaipur', 1600),
                ('potato', 'indore', 1750),
                ('potato', 'hyderabad', 1850),
                ('potato', 'kolkata', 1950),
                ('potato', 'mumbai', 2100),
                ('potato', 'srinagar', 2200),
                ('potato', 'bhubaneswar', 1780),
                ('potato', 'cuttack', 1760),
                ('potato', 'sambalpur', 1720),
                ('potato', 'rourkela', 1820),
                ('potato', 'berhampur', 1880),
                ('potato', 'balasore', 1820),
                ('potato', 'bhadrak', 1800),
                ('potato', 'jeypore', 1740),
                ('potato', 'baripada', 1800),
                ('potato', 'bhawanipatna', 1680),
                ('potato', 'angul', 1750),
                ('potato', 'dhenkanal', 1720),
                ('potato', 'bargarh', 1700),
                ('potato', 'jharsuguda', 1750),
                ('potato', 'koraput', 1850),
                
                # Onion prices
                ('onion', 'delhi', 2500),
                ('onion', 'pune', 2300),
                ('onion', 'jaipur', 2100),
                ('onion', 'bangalore', 2600),
                ('onion', 'hyderabad', 2400),
                ('onion', 'mumbai', 2700),
                ('onion', 'kolkata', 2450),
                ('onion', 'indore', 2200),
                ('onion', 'ludhiana', 2350),
                ('onion', 'srinagar', 2800),
                ('onion', 'bhubaneswar', 2380),
                ('onion', 'cuttack', 2360),
                ('onion', 'sambalpur', 2320),
                ('onion', 'rourkela', 2420),
                ('onion', 'berhampur', 2480),
                ('onion', 'balasore', 2400),
                ('onion', 'bhadrak', 2370),
                ('onion', 'jeypore', 2300),
                ('onion', 'baripada', 2380),
                ('onion', 'bhawanipatna', 2280),
                ('onion', 'angul', 2340),
                ('onion', 'dhenkanal', 2310),
                ('onion', 'bargarh', 2250),
                ('onion', 'jharsuguda', 2300),
                ('onion', 'koraput', 2450),
                
                # Tomato prices
                ('tomato', 'delhi', 4000),
                ('tomato', 'pune', 3800),
                ('tomato', 'bangalore', 4200),
                ('tomato', 'hyderabad', 3600),
                ('tomato', 'mumbai', 4300),
                ('tomato', 'kolkata', 3900),
                ('tomato', 'indore', 3850),
                ('tomato', 'jaipur', 3700),
                ('tomato', 'ludhiana', 4100),
                ('tomato', 'srinagar', 4500),
                ('tomato', 'bhubaneswar', 3950),
                ('tomato', 'cuttack', 3900),
                ('tomato', 'sambalpur', 3850),
                ('tomato', 'rourkela', 4000),
                ('tomato', 'berhampur', 4050),
                ('tomato', 'balasore', 3950),
                ('tomato', 'bhadrak', 3900),
                ('tomato', 'jeypore', 3800),
                ('tomato', 'baripada', 3920),
                ('tomato', 'bhawanipatna', 3800),
                ('tomato', 'angul', 3880),
                ('tomato', 'dhenkanal', 3850),
                ('tomato', 'bargarh', 3750),
                ('tomato', 'jharsuguda', 3850),
                ('tomato', 'koraput', 4000),
                
                # Cotton prices
                ('cotton', 'pune', 5500),
                ('cotton', 'jaipur', 5300),
                ('cotton', 'hyderabad', 5600),
                ('cotton', 'bangalore', 5400),
                ('cotton', 'ludhiana', 5250),
                ('cotton', 'indore', 5350),
                ('cotton', 'mumbai', 5650),
                ('cotton', 'kolkata', 5500),
                ('cotton', 'bhubaneswar', 5420),
                ('cotton', 'cuttack', 5380),
                ('cotton', 'sambalpur', 5320),
                ('cotton', 'rourkela', 5480),
                ('cotton', 'berhampur', 5550),
                ('cotton', 'balasore', 5450),
                ('cotton', 'bhadrak', 5400),
                ('cotton', 'jeypore', 5320),
                ('cotton', 'baripada', 5420),
                ('cotton', 'bhawanipatna', 5280),
                ('cotton', 'angul', 5380),
                ('cotton', 'dhenkanal', 5350),
                ('cotton', 'bargarh', 5280),
                ('cotton', 'jharsuguda', 5350),
                ('cotton', 'koraput', 5500),
                
                # Sugarcane prices
                ('sugarcane', 'pune', 3200),
                ('sugarcane', 'delhi', 3100),
                ('sugarcane', 'bangalore', 3300),
                ('sugarcane', 'hyderabad', 3400),
                ('sugarcane', 'jaipur', 3000),
                ('sugarcane', 'indore', 3150),
                ('sugarcane', 'ludhiana', 3050),
                ('sugarcane', 'mumbai', 3450),
                ('sugarcane', 'bhubaneswar', 3180),
                ('sugarcane', 'cuttack', 3150),
                ('sugarcane', 'sambalpur', 3100),
                ('sugarcane', 'rourkela', 3220),
                ('sugarcane', 'berhampur', 3280),
                ('sugarcane', 'balasore', 3200),
                ('sugarcane', 'bhadrak', 3170),
                ('sugarcane', 'jeypore', 3100),
                ('sugarcane', 'baripada', 3190),
                ('sugarcane', 'bhawanipatna', 3080),
                ('sugarcane', 'angul', 3140),
                ('sugarcane', 'dhenkanal', 3110),
                ('sugarcane', 'bargarh', 3050),
                ('sugarcane', 'jharsuguda', 3120),
                ('sugarcane', 'koraput', 3250),
                
                # Corn prices
                ('corn', 'jaipur', 1950),
                ('corn', 'pune', 2050),
                ('corn', 'bangalore', 2100),
                ('corn', 'indore', 1900),
                ('corn', 'hyderabad', 2000),
                ('corn', 'delhi', 2080),
                ('corn', 'ludhiana', 1850),
                ('corn', 'mumbai', 2150),
                ('corn', 'bhubaneswar', 1980),
                ('corn', 'cuttack', 1950),
                ('corn', 'sambalpur', 1900),
                ('corn', 'rourkela', 2020),
                ('corn', 'berhampur', 2050),
                ('corn', 'balasore', 1980),
                ('corn', 'bhadrak', 1950),
                ('corn', 'jeypore', 1880),
                ('corn', 'baripada', 1980),
                ('corn', 'bhawanipatna', 1880),
                ('corn', 'angul', 1930),
                ('corn', 'dhenkanal', 1900),
                ('corn', 'bargarh', 1850),
                ('corn', 'jharsuguda', 1900),
                ('corn', 'koraput', 2050),
                
                # Soybean prices
                ('soybean', 'indore', 4200),
                ('soybean', 'pune', 4100),
                ('soybean', 'jaipur', 4000),
                ('soybean', 'bangalore', 4300),
                ('soybean', 'hyderabad', 4150),
                ('soybean', 'mumbai', 4250),
                
                # Chickpea prices
                ('chickpea', 'indore', 5200),
                ('chickpea', 'jaipur', 5100),
                ('chickpea', 'pune', 5300),
                ('chickpea', 'bangalore', 5400),
                ('chickpea', 'delhi', 5250),
                ('chickpea', 'hyderabad', 5350),
                
                # Lentils prices
                ('lentils', 'pune', 6500),
                ('lentils', 'bangalore', 6300),
                ('lentils', 'hyderabad', 6700),
                ('lentils', 'kolkata', 6400),
                ('lentils', 'delhi', 6600),
                ('lentils', 'mumbai', 6800),
                
                # Pepper prices
                ('pepper', 'kolkata', 35000),
                ('pepper', 'bangalore', 34000),
                ('pepper', 'hyderabad', 36000),
                ('pepper', 'mumbai', 37000),
                
                # Garlic prices
                ('garlic', 'pune', 8000),
                ('garlic', 'indore', 7800),
                ('garlic', 'jaipur', 8200),
                ('garlic', 'delhi', 8100),
                ('garlic', 'hyderabad', 8300),
                
                # Ginger prices
                ('ginger', 'pune', 12000),
                ('ginger', 'srinagar', 13000),
                ('ginger', 'kolkata', 11000),
                ('ginger', 'bangalore', 12500),
                ('ginger', 'mumbai', 12800),
                
                # Turmeric prices
                ('turmeric', 'hyderabad', 9000),
                ('turmeric', 'pune', 8800),
                ('turmeric', 'bangalore', 9200),
                ('turmeric', 'indore', 8900),
                ('turmeric', 'mumbai', 9300),
            ]
            
            added_count = 0
            for crop_id, mandi_id, price in sample_prices:
                self.db.add_price(crop_id, mandi_id, price)
                added_count += 1
            
            logger.info(f"✓ Loaded {added_count} sample price entries")
            
            return {
                'success': True,
                'message': f'Loaded {added_count} sample price entries',
                'count': added_count
            }
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }