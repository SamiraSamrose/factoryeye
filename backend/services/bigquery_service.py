"""
BigQuery Service
Manages data storage and queries for sensor readings, anomalies, and predictions
Provides interface to Google BigQuery
"""

from google.cloud import bigquery
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class BigQueryService:
    """
    Service for BigQuery operations
    Handles all database interactions for analytical storage
    """
    
    def __init__(self):
        self.client = bigquery.Client(project=settings.project_id)
        self.dataset_id = settings.bigquery_dataset
        
        # Table references
        self.readings_table = f"{settings.project_id}.{self.dataset_id}.{settings.bigquery_table_readings}"
        self.anomalies_table = f"{settings.project_id}.{self.dataset_id}.{settings.bigquery_table_anomalies}"
        self.predictions_table = f"{settings.project_id}.{self.dataset_id}.{settings.bigquery_table_predictions}"
        self.metrics_table = f"{settings.project_id}.{self.dataset_id}.{settings.bigquery_table_metrics}"
        
        logger.info(f"BigQueryService initialized: dataset={self.dataset_id}")
    
    async def insert_sensor_reading(self, data: Dict[str, Any]):
        """
        Insert sensor reading into BigQuery
        
        Steps:
        1. Validate data schema
        2. Format for BigQuery
        3. Insert row
        
        Args:
            data: Sensor reading data
        """
        try:
            rows_to_insert = [{
                "sensor_id": data["sensor_id"],
                "timestamp": data["timestamp"],
                "machine_id": data["machine_id"],
                "metrics": data["metrics"]
            }]
            
            errors = self.client.insert_rows_json(
                self.readings_table,
                rows_to_insert
            )
            
            if errors:
                logger.error(f"Errors inserting reading: {errors}")
            else:
                logger.debug(f"Inserted reading: sensor={data['sensor_id']}")
                
        except Exception as e:
            logger.error(f"Error inserting sensor reading: {e}")
            raise
    
    async def insert_anomaly(self, data: Dict[str, Any]):
        """
        Insert anomaly record into BigQuery
        
        Steps:
        1. Format anomaly data
        2. Insert row with metadata
        3. Log insertion
        
        Args:
            data: Anomaly data including score and severity
        """
        try:
            rows_to_insert = [{
                "sensor_id": data["sensor_id"],
                "timestamp": data["timestamp"],
                "anomaly_score": data["anomaly_score"],
                "severity": data["severity"],
                "metrics": data["metrics"],
                "detected_at": data["detected_at"]
            }]
            
            errors = self.client.insert_rows_json(
                self.anomalies_table,
                rows_to_insert
            )
            
            if errors:
                logger.error(f"Errors inserting anomaly: {errors}")
            else:
                logger.debug(
                    f"Inserted anomaly: sensor={data['sensor_id']}, "
                    f"score={data['anomaly_score']}"
                )
                
        except Exception as e:
            logger.error(f"Error inserting anomaly: {e}")
            raise
    
    async def insert_predictions(self, predictions: List[Dict[str, Any]]):
        """
        Insert batch of RUL predictions into BigQuery
        
        Steps:
        1. Format prediction batch
        2. Batch insert rows
        3. Handle errors
        
        Args:
            predictions: List of prediction dictionaries
        """
        try:
            rows_to_insert = []
            
            for pred in predictions:
                rows_to_insert.append({
                    "machine_id": pred["machine_id"],
                    "rul_hours": pred["rul_hours"],
                    "confidence": pred["confidence"],
                    "priority": pred["priority"],
                    "failure_modes": pred["failure_modes"],
                    "recommended_maintenance_date": pred["recommended_maintenance_date"],
                    "predicted_at": pred["predicted_at"]
                })
            
            errors = self.client.insert_rows_json(
                self.predictions_table,
                rows_to_insert
            )
            
            if errors:
                logger.error(f"Errors inserting predictions: {errors}")
            else:
                logger.info(f"Inserted {len(predictions)} predictions")
                
        except Exception as e:
            logger.error(f"Error inserting predictions: {e}")
            raise
    
    async def query_to_dataframe(
        self,
        query: str,
        job_config: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame
        
        Steps:
        1. Execute query with optional parameters
        2. Wait for completion
        3. Convert to pandas DataFrame
        4. Return results
        
        Args:
            query: SQL query string
            job_config: Optional query configuration
            
        Returns:
            Pandas DataFrame with query results
        """
        try:
            # Configure query job
            config = bigquery.QueryJobConfig()
            if job_config and "query_parameters" in job_config:
                config.query_parameters = [
                    bigquery.ScalarQueryParameter(
                        param["name"],
                        param["parameterType"]["type"],
                        param["parameterValue"]["value"]
                    )
                    for param in job_config["query_parameters"]
                ]
            
            # Execute query
            query_job = self.client.query(query, job_config=config)
            
            # Wait for completion and convert to DataFrame
            df = query_job.to_dataframe()
            
            logger.debug(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    async def get_recent_anomalies(
        self,
        hours: int = 24,
        machine_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query recent anomalies
        
        Steps:
        1. Build query with time filter
        2. Execute query
        3. Return results as list
        
        Args:
            hours: Hours to look back
            machine_id: Optional machine filter
            
        Returns:
            List of anomaly records
        """
        machine_filter = ""
        if machine_id:
            machine_filter = f"AND sr.machine_id = '{machine_id}'"
        
        query = f"""
        SELECT
            a.sensor_id,
            a.timestamp,
            a.anomaly_score,
            a.severity,
            a.metrics,
            sr.machine_id
        FROM
            `{self.anomalies_table}` a
        JOIN
            `{self.readings_table}` sr
        ON
            a.sensor_id = sr.sensor_id
            AND a.timestamp = sr.timestamp
        WHERE
            a.timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
            {machine_filter}
        ORDER BY
            a.timestamp DESC
        LIMIT 1000
        """
        
        df = await self.query_to_dataframe(query)
        return df.to_dict(orient='records')