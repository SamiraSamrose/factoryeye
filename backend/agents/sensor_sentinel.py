"""
Agent 1: Sensor Sentinel
Real-time anomaly detection on streaming sensor data using Gemma on GPU
Processes Pub/Sub messages and detects anomalies in sensor readings
"""

import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Any
import torch

from backend.config import settings
from backend.services.bigquery_service import BigQueryService
from backend.services.firestore_service import FirestoreService
from backend.models.anomaly_detector import AnomalyDetector

logger = logging.getLogger(__name__)


class SensorSentinel:
    """
    Real-time anomaly detection agent
    Uses sliding window analysis and ML-based detection with Gemma model
    """
    
    def __init__(self):
        self.bigquery = BigQueryService()
        self.firestore = FirestoreService()
        self.anomaly_detector = AnomalyDetector()
        self.window_size = 30  # 30-second sliding window
        self.sensor_buffers = {}  # Store recent readings per sensor
        
        logger.info("Sensor Sentinel initialized")
    
    async def process_message(self, message: Any) -> Dict[str, Any]:
        """
        Process incoming sensor message from Pub/Sub
        
        Steps:
        1. Parse and validate message
        2. Add to sliding window buffer
        3. Run anomaly detection
        4. Store results in BigQuery
        5. Trigger alerts if anomaly detected
        
        Args:
            message: Pub/Sub message containing sensor data
            
        Returns:
            Dict containing processing results and anomaly flag
        """
        try:
            # Parse message data
            data = self._parse_message(message)
            sensor_id = data["sensor_id"]
            timestamp = data["timestamp"]
            metrics = data["metrics"]
            
            # Update sensor buffer with new reading
            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = []
            
            self.sensor_buffers[sensor_id].append({
                "timestamp": timestamp,
                "metrics": metrics
            })
            
            # Maintain window size
            if len(self.sensor_buffers[sensor_id]) > self.window_size:
                self.sensor_buffers[sensor_id].pop(0)
            
            # Run anomaly detection if we have enough data
            is_anomaly = False
            anomaly_score = 0.0
            
            if len(self.sensor_buffers[sensor_id]) >= 10:
                is_anomaly, anomaly_score = await self._detect_anomaly(
                    sensor_id,
                    self.sensor_buffers[sensor_id]
                )
            
            # Store raw reading in BigQuery
            await self.bigquery.insert_sensor_reading(data)
            
            # Store anomaly if detected
            if is_anomaly:
                await self._handle_anomaly(
                    sensor_id,
                    timestamp,
                    metrics,
                    anomaly_score
                )
            
            return {
                "sensor_id": sensor_id,
                "timestamp": timestamp,
                "is_anomaly": is_anomaly,
                "anomaly_score": anomaly_score,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise
    
    def _parse_message(self, message: Any) -> Dict[str, Any]:
        """
        Parse and validate Pub/Sub message
        
        Steps:
        1. Decode message data
        2. Validate schema
        3. Extract sensor readings
        
        Args:
            message: Raw Pub/Sub message
            
        Returns:
            Parsed sensor data dictionary
        """
        import json
        
        data = json.loads(message.data.decode('utf-8'))
        
        # Validate required fields
        required_fields = ["sensor_id", "timestamp", "machine_id", "metrics"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        return data
    
    async def _detect_anomaly(
        self,
        sensor_id: str,
        readings: List[Dict[str, Any]]
    ) -> tuple[bool, float]:
        """
        Detect anomalies in sensor readings using Gemma model
        
        Steps:
        1. Extract feature vectors from readings
        2. Normalize features
        3. Run Gemma inference on GPU
        4. Calculate anomaly score
        5. Apply threshold
        
        Args:
            sensor_id: Sensor identifier
            readings: List of recent sensor readings
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        try:
            # Extract features from readings
            features = self._extract_features(readings)
            
            # Get sensor configuration from Firestore
            config = await self.firestore.get_sensor_config(sensor_id)
            
            # Run anomaly detection model
            anomaly_score = await self.anomaly_detector.detect(
                features,
                config
            )
            
            # Check against threshold
            is_anomaly = anomaly_score >= settings.anomaly_threshold
            
            logger.debug(
                f"Sensor {sensor_id}: anomaly_score={anomaly_score:.3f}, "
                f"is_anomaly={is_anomaly}"
            )
            
            return is_anomaly, float(anomaly_score)
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return False, 0.0
    
    def _extract_features(self, readings: List[Dict[str, Any]]) -> np.ndarray:
        """
        Extract feature vectors from sensor readings
        
        Steps:
        1. Extract metric values
        2. Calculate statistical features (mean, std, min, max)
        3. Calculate temporal features (rate of change, trends)
        4. Combine into feature vector
        
        Args:
            readings: List of sensor readings
            
        Returns:
            Feature vector as numpy array
        """
        # Extract metric time series
        temperature = [r["metrics"].get("temperature", 0) for r in readings]
        vibration = [r["metrics"].get("vibration", 0) for r in readings]
        pressure = [r["metrics"].get("pressure", 0) for r in readings]
        rpm = [r["metrics"].get("rpm", 0) for r in readings]
        
        # Calculate statistical features
        features = []
        for series in [temperature, vibration, pressure, rpm]:
            features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.percentile(series, 25),
                np.percentile(series, 75)
            ])
        
        # Calculate rate of change
        for series in [temperature, vibration, pressure, rpm]:
            if len(series) > 1:
                changes = np.diff(series)
                features.extend([
                    np.mean(changes),
                    np.std(changes)
                ])
            else:
                features.extend([0, 0])
        
        return np.array(features)
    
    async def _handle_anomaly(
        self,
        sensor_id: str,
        timestamp: str,
        metrics: Dict[str, float],
        anomaly_score: float
    ):
        """
        Handle detected anomaly
        
        Steps:
        1. Store anomaly record in BigQuery
        2. Check alert rules from Firestore
        3. Determine severity level
        4. Trigger alerts if needed
        
        Args:
            sensor_id: Sensor identifier
            timestamp: Time of anomaly
            metrics: Sensor metrics at time of anomaly
            anomaly_score: Calculated anomaly score
        """
        # Determine severity based on score
        if anomaly_score >= 0.95:
            severity = "critical"
        elif anomaly_score >= 0.90:
            severity = "high"
        elif anomaly_score >= 0.85:
            severity = "medium"
        else:
            severity = "low"
        
        # Store anomaly in BigQuery
        await self.bigquery.insert_anomaly({
            "sensor_id": sensor_id,
            "timestamp": timestamp,
            "anomaly_score": anomaly_score,
            "severity": severity,
            "metrics": metrics,
            "detected_at": datetime.utcnow().isoformat()
        })
        
        logger.warning(
            f"Anomaly detected: sensor={sensor_id}, "
            f"severity={severity}, score={anomaly_score:.3f}"
        )