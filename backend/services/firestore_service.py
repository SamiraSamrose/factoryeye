"""
Firestore Service
Manages sensor configurations and application state
Provides interface to Google Firestore
"""

from google.cloud import firestore
import logging
from typing import Dict, List, Any, Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class FirestoreService:
    """
    Service for Firestore operations
    Handles sensor configurations, alert rules, and application state
    """
    
    def __init__(self):
        self.db = firestore.Client(project=settings.project_id)
        self.sensors_collection = settings.firestore_collection
        self.alerts_collection = settings.firestore_alerts_collection
        
        logger.info("FirestoreService initialized")
    
    async def get_sensor_config(self, sensor_id: str) -> Optional[Dict[str, Any]]:
        """
        Get sensor configuration from Firestore
        
        Steps:
        1. Query sensor document by ID
        2. Return configuration data
        3. Handle missing sensors
        
        Args:
            sensor_id: Sensor identifier
            
        Returns:
            Sensor configuration dict or None
        """
        try:
            doc_ref = self.db.collection(self.sensors_collection).document(sensor_id)
            doc = doc_ref.get()
            
            if doc.exists:
                config = doc.to_dict()
                logger.debug(f"Retrieved config for sensor: {sensor_id}")
                return config
            else:
                logger.warning(f"Sensor config not found: {sensor_id}")
                # Return default configuration
                return {
                    "sensor_id": sensor_id,
                    "active": True,
                    "thresholds": {
                        "temperature": {"min": 20, "max": 85},
                        "vibration": {"min": 0, "max": 50},
                        "pressure": {"min": 1, "max": 10},
                        "rpm": {"min": 0, "max": 5000}
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting sensor config: {e}")
            return None
    
    async def get_all_sensor_configs(self) -> List[Dict[str, Any]]:
        """
        Get all sensor configurations
        
        Steps:
        1. Query all documents in sensors collection
        2. Convert to list of dictionaries
        3. Return results
        
        Returns:
            List of sensor configuration dicts
        """
        try:
            configs = []
            docs = self.db.collection(self.sensors_collection).stream()
            
            for doc in docs:
                config = doc.to_dict()
                config["sensor_id"] = doc.id
                configs.append(config)
            
            logger.debug(f"Retrieved {len(configs)} sensor configs")
            return configs
            
        except Exception as e:
            logger.error(f"Error getting all sensor configs: {e}")
            return []
    
    async def update_sensor_config(
        self,
        sensor_id: str,
        config: Dict[str, Any]
    ):
        """
        Update sensor configuration
        
        Steps:
        1. Reference sensor document
        2. Update or create document
        3. Log update
        
        Args:
            sensor_id: Sensor identifier
            config: Configuration data to update
        """
        try:
            doc_ref = self.db.collection(self.sensors_collection).document(sensor_id)
            doc_ref.set(config, merge=True)
            
            logger.info(f"Updated sensor config: {sensor_id}")
            
        except Exception as e:
            logger.error(f"Error updating sensor config: {e}")
            raise
    
    async def update_maintenance_schedule(self, schedule: Dict[str, Any]):
        """
        Update maintenance schedule in Firestore
        
        Steps:
        1. Store schedule in dedicated document
        2. Include timestamp
        3. Log update
        
        Args:
            schedule: Maintenance schedule data
        """
        try:
            doc_ref = self.db.collection("maintenance_schedules").document("current")
            doc_ref.set(schedule)
            
            logger.info("Updated maintenance schedule")
            
        except Exception as e:
            logger.error(f"Error updating maintenance schedule: {e}")
            raise
    
    async def get_alert_rules(self) -> List[Dict[str, Any]]:
        """
        Get alert rules from Firestore
        
        Steps:
        1. Query alert rules collection
        2. Return list of active rules
        
        Returns:
            List of alert rule dicts
        """
        try:
            rules = []
            docs = self.db.collection(self.alerts_collection).stream()
            
            for doc in docs:
                rule = doc.to_dict()
                rule["rule_id"] = doc.id
                if rule.get("active", True):
                    rules.append(rule)
            
            logger.debug(f"Retrieved {len(rules)} alert rules")
            return rules
            
        except Exception as e:
            logger.error(f"Error getting alert rules: {e}")
            return []