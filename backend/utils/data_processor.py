"""
Data Processing Utilities
Common data processing and transformation functions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def normalize_sensor_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize sensor data to standard format
    
    Steps:
    1. Validate required fields
    2. Convert timestamp to ISO format
    3. Normalize metric values
    4. Add metadata
    
    Args:
        data: Raw sensor data
        
    Returns:
        Normalized sensor data
    """
    normalized = {
        "sensor_id": data.get("sensor_id"),
        "machine_id": data.get("machine_id"),
        "timestamp": data.get("timestamp"),
        "metrics": {}
    }
    
    # Normalize metrics
    metrics = data.get("metrics", {})
    normalized["metrics"] = {
        "temperature": float(metrics.get("temperature", 0)),
        "vibration": float(metrics.get("vibration", 0)),
        "pressure": float(metrics.get("pressure", 0)),
        "rpm": float(metrics.get("rpm", 0))
    }
    
    return normalized


def aggregate_time_series(
    data: List[Dict[str, Any]],
    interval_minutes: int = 5
) -> pd.DataFrame:
    """
    Aggregate time series data by interval
    
    Steps:
    1. Convert to DataFrame
    2. Set timestamp index
    3. Resample by interval
    4. Aggregate metrics
    
    Args:
        data: List of sensor readings
        interval_minutes: Aggregation interval
        
    Returns:
        Aggregated DataFrame
    """
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    
    # Resample and aggregate
    aggregated = df.resample(f"{interval_minutes}T").agg({
        "temperature": "mean",
        "vibration": "mean",
        "pressure": "mean",
        "rpm": "mean"
    })
    
    return aggregated


def calculate_moving_average(
    values: List[float],
    window_size: int = 10
) -> List[float]:
    """
    Calculate moving average
    
    Args:
        values: Time series values
        window_size: Window size for moving average
        
    Returns:
        List of moving average values
    """
    if len(values) < window_size:
        return values
    
    ma = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    
    # Pad beginning to maintain length
    padding = [values[0]] * (window_size - 1)
    return padding + ma.tolist()


def detect_outliers(values: List[float], threshold: float = 3.0) -> List[int]:
    """
    Detect outliers using z-score method
    
    Steps:
    1. Calculate mean and standard deviation
    2. Compute z-scores
    3. Identify values exceeding threshold
    
    Args:
        values: Time series values
        threshold: Z-score threshold
        
    Returns:
        List of outlier indices
    """
    if len(values) < 2:
        return []
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array)
    
    if std == 0:
        return []
    
    z_scores = np.abs((values_array - mean) / std)
    outlier_indices = np.where(z_scores > threshold)[0].tolist()
    
    return outlier_indices