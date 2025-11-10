"""
API Routes
REST endpoints for dashboard and external integrations
Provides access to sensors, anomalies, predictions, and analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
import logging

from backend.services.bigquery_service import BigQueryService
from backend.services.firestore_service import FirestoreService
from backend.agents.efficiency_analyst import EfficiencyAnalyst

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
bigquery_service = BigQueryService()
firestore_service = FirestoreService()
efficiency_analyst = EfficiencyAnalyst()


@router.get("/sensors")
async def list_sensors():
    """
    List all active sensors
    
    Returns sensor configurations from Firestore
    
    Returns:
        List of sensor configuration objects
    """
    try:
        sensors = await firestore_service.get_all_sensor_configs()
        return {
            "count": len(sensors),
            "sensors": sensors
        }
    except Exception as e:
        logger.error(f"Error listing sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sensors/{sensor_id}")
async def get_sensor(sensor_id: str):
    """
    Get specific sensor configuration
    
    Args:
        sensor_id: Sensor identifier
        
    Returns:
        Sensor configuration object
    """
    try:
        config = await firestore_service.get_sensor_config(sensor_id)
        if not config:
            raise HTTPException(status_code=404, detail="Sensor not found")
        return config
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sensor: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def list_anomalies(
    hours: int = Query(default=24, ge=1, le=168),
    machine_id: Optional[str] = None
):
    """
    List recent anomalies
    
    Queries BigQuery for anomalies within time range
    
    Args:
        hours: Hours to look back (1-168)
        machine_id: Optional machine filter
        
    Returns:
        List of anomaly records
    """
    try:
        anomalies = await bigquery_service.get_recent_anomalies(hours, machine_id)
        return {
            "count": len(anomalies),
            "time_range_hours": hours,
            "machine_id": machine_id,
            "anomalies": anomalies
        }
    except Exception as e:
        logger.error(f"Error listing anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions")
async def list_predictions(machine_id: Optional[str] = None):
    """
    List RUL predictions
    
    Queries BigQuery for latest predictions
    
    Args:
        machine_id: Optional machine filter
        
    Returns:
        List of prediction records
    """
    try:
        machine_filter = ""
        if machine_id:
            machine_filter = f"WHERE machine_id = '{machine_id}'"
        
        query = f"""
        SELECT
            machine_id,
            rul_hours,
            confidence,
            priority,
            failure_modes,
            recommended_maintenance_date,
            predicted_at
        FROM
            `{bigquery_service.predictions_table}`
        {machine_filter}
        ORDER BY
            predicted_at DESC
        LIMIT 100
        """
        
        df = await bigquery_service.query_to_dataframe(query)
        predictions = df.to_dict(orient='records')
        
        return {
            "count": len(predictions),
            "machine_id": machine_id,
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Error listing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_efficiency(
    machine_id: Optional[str] = None,
    time_range_hours: int = Query(default=24, ge=1, le=168)
):
    """
    Request efficiency analysis
    
    Triggers Efficiency Analyst agent to generate insights and visualizations
    
    Args:
        machine_id: Optional machine filter
        time_range_hours: Hours to analyze (1-168)
        
    Returns:
        Analysis report with insights and visualizations
    """
    try:
        report = await efficiency_analyst.analyze_efficiency(
            machine_id,
            time_range_hours
        )
        return report
    except Exception as e:
        logger.error(f"Error in efficiency analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """
    Get dashboard statistics
    
    Queries aggregated metrics for dashboard overview
    
    Returns:
        Dashboard statistics object
    """
    try:
        # Query overall statistics
        query = f"""
        WITH recent_data AS (
            SELECT
                COUNT(DISTINCT machine_id) as total_machines,
                COUNT(DISTINCT sensor_id) as total_sensors,
                COUNT(*) as total_readings
            FROM
                `{bigquery_service.readings_table}`
            WHERE
                timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ),
        recent_anomalies AS (
            SELECT
                COUNT(*) as total_anomalies,
                AVG(anomaly_score) as avg_anomaly_score
            FROM
                `{bigquery_service.anomalies_table}`
            WHERE
                timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ),
        urgent_predictions AS (
            SELECT
                COUNT(*) as urgent_maintenance_count
            FROM
                `{bigquery_service.predictions_table}`
            WHERE
                priority = 'urgent'
                AND predicted_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        )
        SELECT
            rd.total_machines,
            rd.total_sensors,
            rd.total_readings,
            ra.total_anomalies,
            ra.avg_anomaly_score,
            up.urgent_maintenance_count
        FROM
            recent_data rd,
            recent_anomalies ra,
            urgent_predictions up
        """
        
        df = await bigquery_service.query_to_dataframe(query)
        stats = df.to_dict(orient='records')[0] if len(df) > 0 else {}
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/machines")
async def list_machines():
    """
    List all machines
    
    Queries distinct machine IDs from sensor configurations
    
    Returns:
        List of machine objects with status
    """
    try:
        configs = await firestore_service.get_all_sensor_configs()
        machines = {}
        
        for config in configs:
            machine_id = config.get("machine_id")
            if machine_id and machine_id not in machines:
                machines[machine_id] = {
                    "machine_id": machine_id,
                    "active": config.get("active", True),
                    "sensor_count": 0
                }
            if machine_id:
                machines[machine_id]["sensor_count"] += 1
        
        return {
            "count": len(machines),
            "machines": list(machines.values())
        }
    except Exception as e:
        logger.error(f"Error listing machines: {e}")
        raise HTTPException(status_code=500, detail=str(e))