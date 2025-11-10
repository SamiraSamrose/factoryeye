"""
Data Validation Utilities
Validates incoming data against schemas
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def validate_sensor_reading(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate sensor reading data
    
    Steps:
    1. Check required fields present
    2. Validate data types
    3. Check value ranges
    
    Args:
        data: Sensor reading data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["sensor_id", "timestamp", "machine_id", "metrics"]
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate sensor_id
    if not isinstance(data["sensor_id"], str) or not data["sensor_id"]:
        return False, "Invalid sensor_id"
    
    # Validate machine_id
    if not isinstance(data["machine_id"], str) or not data["machine_id"]:
        return False, "Invalid machine_id"
    
    # Validate metrics
    metrics = data.get("metrics", {})
    if not isinstance(metrics, dict):
        return False, "Invalid metrics format"
    
    required_metrics = ["temperature", "vibration", "pressure", "rpm"]
    for metric in required_metrics:
        if metric not in metrics:
            return False, f"Missing metric: {metric}"
        
        value = metrics[metric]
        if not isinstance(value, (int, float)):
            return False, f"Invalid metric value type: {metric}"
        
        # Range validation
        if metric == "temperature" and not (-50 <= value <= 200):
            return False, f"Temperature out of range: {value}"
        elif metric == "vibration" and not (0 <= value <= 100):
            return False, f"Vibration out of range: {value}"
        elif metric == "pressure" and not (0 <= value <= 20):
            return False, f"Pressure out of range: {value}"
        elif metric == "rpm" and not (0 <= value <= 10000):
            return False, f"RPM out of range: {value}"
    
    return True, ""


def validate_anomaly_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate anomaly data
    
    Args:
        data: Anomaly data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["sensor_id", "timestamp", "anomaly_score", "severity"]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate anomaly_score
    score = data.get("anomaly_score")
    if not isinstance(score, (int, float)) or not (0 <= score <= 1):
        return False, "Invalid anomaly_score (must be 0-1)"
    
    # Validate severity
    severity = data.get("severity")
    valid_severities = ["low", "medium", "high", "critical"]
    if severity not in valid_severities:
        return False, f"Invalid severity (must be one of {valid_severities})"
    
    return True, ""