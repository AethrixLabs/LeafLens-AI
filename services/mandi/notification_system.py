"""
Price Alert Notification System
Manages price alerts and notifications
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationSystem:
    """Manages price alerts and notifications"""
    
    def __init__(self):
        self.alerts_file = "data/price_alerts.json"
        self.ensure_alerts_file()
    
    def ensure_alerts_file(self):
        """Create alerts file if it doesn't exist"""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.alerts_file):
            initial_data = {'alerts': []}
            with open(self.alerts_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
    
    def set_alert(self, crop_id: str, target_price: float, alert_type: str = 'above') -> Dict:
        """
        Set a price alert
        alert_type: 'above' or 'below'
        """
        try:
            logger.info(f"Setting alert: {crop_id} - Price {alert_type} Rs.{target_price}")
            
            data = self._read_alerts()
            
            alert = {
                'id': len(data['alerts']) + 1,
                'crop_id': crop_id,
                'target_price': target_price,
                'alert_type': alert_type,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'triggered': False
            }
            
            data['alerts'].append(alert)
            self._write_alerts(data)
            
            logger.info(f"✓ Alert created: {alert['id']}")
            
            return {
                'success': True,
                'alert': alert,
                'message': f'Alert set for {crop_id}: Price {alert_type} Rs.{target_price}'
            }
        except Exception as e:
            logger.error(f"Error setting alert: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_all_alerts(self) -> List[Dict]:
        """Get all active alerts"""
        data = self._read_alerts()
        return [a for a in data['alerts'] if a['status'] == 'active']
    
    def check_alerts(self, crop_id: str, current_price: float) -> List[Dict]:
        """Check if any alerts are triggered for a crop"""
        data = self._read_alerts()
        triggered = []
        
        for alert in data['alerts']:
            if alert['crop_id'] != crop_id or alert['status'] != 'active':
                continue
            
            # Check if alert condition is met
            if alert['alert_type'] == 'above' and current_price >= alert['target_price']:
                triggered.append({
                    **alert,
                    'message': f"Price alert! {crop_id} is now Rs.{current_price} (target: Rs.{alert['target_price']})",
                    'triggered_at': datetime.now().isoformat()
                })
                # Mark as triggered
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
            
            elif alert['alert_type'] == 'below' and current_price <= alert['target_price']:
                triggered.append({
                    **alert,
                    'message': f"Price alert! {crop_id} is now Rs.{current_price} (target: Rs.{alert['target_price']})",
                    'triggered_at': datetime.now().isoformat()
                })
                # Mark as triggered
                alert['triggered'] = True
                alert['triggered_at'] = datetime.now().isoformat()
        
        # Update alerts file with triggered status
        if triggered:
            self._write_alerts(data)
        
        return triggered
    
    def delete_alert(self, alert_id: int) -> Dict:
        """Delete an alert"""
        try:
            data = self._read_alerts()
            
            alert_found = False
            for alert in data['alerts']:
                if alert['id'] == alert_id:
                    alert['status'] = 'deleted'
                    alert_found = True
                    break
            
            if alert_found:
                self._write_alerts(data)
                logger.info(f"✓ Alert {alert_id} deleted")
                return {
                    'success': True,
                    'message': f'Alert {alert_id} deleted'
                }
            else:
                return {
                    'success': False,
                    'message': f'Alert {alert_id} not found'
                }
        except Exception as e:
            logger.error(f"Error deleting alert: {str(e)}")
            return {
                'success': False,
                'message': str(e)
            }
    
    def get_triggered_alerts(self) -> List[Dict]:
        """Get all triggered alerts"""
        data = self._read_alerts()
        return [a for a in data['alerts'] if a.get('triggered', False)]
    
    def _read_alerts(self) -> Dict:
        """Read alerts file"""
        with open(self.alerts_file, 'r') as f:
            return json.load(f)
    
    def _write_alerts(self, data: Dict):
        """Write alerts file"""
        with open(self.alerts_file, 'w') as f:
            json.dump(data, f, indent=2)