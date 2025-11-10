"""
Agent 2: Predictor AI
Batch predictive maintenance using Gemma on GPU
Predicts equipment failures and calculates Remaining Useful Life (RUL)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import torch

from backend.config import settings
from backend.services.bigquery_service import BigQueryService
from backend.services.firestore_service import FirestoreService
from backend.models.predictive_model import PredictiveModel

logger = logging.getLogger(__name__)


class PredictorAI:
    """
    Predictive maintenance agent
    Analyzes historical data to forecast equipment failures and RUL
    Runs as scheduled Cloud Run Job (hourly execution)
    """
    
    def __init__(self):
        self.bigquery = BigQueryService()
        self.firestore = FirestoreService()
        self.predictive_model = PredictiveModel()
        self.prediction_window = settings.prediction_window_hours
        
        logger.info("Predictor AI initialized")
    
    async def run_prediction_job(self):
        """
        Main prediction job execution
        
        Steps:
        1. Query historical sensor data from BigQuery
        2. Aggregate data by machine
        3. Extract degradation patterns
        4. Run Gemma-based RUL prediction
        5. Generate maintenance schedules
        6. Store predictions in BigQuery
        
        This method is called by Cloud Run Job scheduler
        """
        logger.info("Starting prediction job...")
        
        try:
            # Get list of active machines
            machines = await self._get_active_machines()
            logger.info(f"Processing {len(machines)} machines")
            
            predictions = []
            
            for machine_id in machines:
                try:
                    # Get historical data for machine
                    historical_data = await self._get_machine_history(
                        machine_id
                    )
                    
                    if len(historical_data) < 100:
                        logger.warning(
                            f"Insufficient data for machine {machine_id}"
                        )
                        continue
                    
                    # Run RUL prediction
                    prediction = await self._predict_rul(
                        machine_id,
                        historical_data
                    )
                    
                    predictions.append(prediction)
                    
                    logger.info(
                        f"Machine {machine_id}: RUL={prediction['rul_hours']:.1f}h, "
                        f"confidence={prediction['confidence']:.2f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing machine {machine_id}: {e}")
                    continue
            
            # Store all predictions in BigQuery
            if predictions:
                await self.bigquery.insert_predictions(predictions)
            
            # Generate maintenance schedule
            await self._generate_maintenance_schedule(predictions)
            
            logger.info(
                f"Prediction job completed. Processed {len(predictions)} machines."
            )
            
        except Exception as e:
            logger.error(f"Error in prediction job: {e}")
            raise
    
    async def _get_active_machines(self) -> List[str]:
        """
        Query active machines from Firestore
        
        Steps:
        1. Query sensor_configs collection
        2. Extract unique machine IDs
        3. Filter active machines
        
        Returns:
            List of active machine IDs
        """
        configs = await self.firestore.get_all_sensor_configs()
        machines = set()
        
        for config in configs:
            if config.get("active", False):
                machine_id = config.get("machine_id")
                if machine_id:
                    machines.add(machine_id)
        
        return list(machines)
    
    async def _get_machine_history(
        self,
        machine_id: str
    ) -> pd.DataFrame:
        """
        Query historical sensor data for a machine
        
        Steps:
        1. Query BigQuery for last N days of data
        2. Filter by machine_id
        3. Aggregate sensor readings
        4. Sort by timestamp
        
        Args:
            machine_id: Machine identifier
            
        Returns:
            DataFrame with historical sensor readings
        """
        query = f"""
        SELECT
            timestamp,
            sensor_id,
            metrics.temperature,
            metrics.vibration,
            metrics.pressure,
            metrics.rpm
        FROM
            `{settings.project_id}.{settings.bigquery_dataset}.{settings.bigquery_table_readings}`
        WHERE
            machine_id = @machine_id
            AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY
            timestamp ASC
        """
        
        job_config = {
            "query_parameters": [
                {
                    "name": "machine_id",
                    "parameterType": {"type": "STRING"},
                    "parameterValue": {"value": machine_id}
                }
            ]
        }
        
        df = await self.bigquery.query_to_dataframe(query, job_config)
        return df
    
    async def _predict_rul(
        self,
        machine_id: str,
        historical_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict Remaining Useful Life using Gemma model
        
        Steps:
        1. Preprocess historical data
        2. Extract degradation features
        3. Run Gemma inference on GPU
        4. Calculate RUL estimate
        5. Determine confidence level
        6. Identify failure modes
        
        Args:
            machine_id: Machine identifier
            historical_data: Historical sensor readings
            
        Returns:
            Dict with RUL prediction and metadata
        """
        # Preprocess data
        features = self._extract_degradation_features(historical_data)
        
        # Run predictive model
        rul_hours, confidence, failure_modes = await self.predictive_model.predict(
            features
        )
        
        # Determine maintenance priority
        if rul_hours < 24:
            priority = "urgent"
        elif rul_hours < 168:  # 1 week
            priority = "high"
        elif rul_hours < 720:  # 30 days
            priority = "medium"
        else:
            priority = "low"
        
        # Calculate recommended maintenance date
        maintenance_date = datetime.utcnow() + timedelta(hours=float(rul_hours))
        
        return {
            "machine_id": machine_id,
            "rul_hours": float(rul_hours),
            "confidence": float(confidence),
            "priority": priority,
            "failure_modes": failure_modes,
            "recommended_maintenance_date": maintenance_date.isoformat(),
            "predicted_at": datetime.utcnow().isoformat()
        }
    
    def _extract_degradation_features(
        self,
        historical_data: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract features indicating equipment degradation
        
        Steps:
        1. Calculate trend lines for each metric
        2. Identify increasing/decreasing patterns
        3. Calculate variance over time
        4. Extract cycle count features
        5. Identify operating regime changes
        
        Args:
            historical_data: Historical sensor readings
            
        Returns:
            Feature vector for RUL prediction
        """
        features = []
        
        # Calculate trends for each metric
        for column in ['temperature', 'vibration', 'pressure', 'rpm']:
            if column in historical_data.columns:
                values = historical_data[column].values
                
                # Linear trend
                if len(values) > 1:
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    features.append(slope)
                else:
                    features.append(0)
                
                # Variance trend (increasing variance indicates degradation)
                window_size = 50
                if len(values) > window_size:
                    variances = [
                        np.var(values[i:i+window_size])
                        for i in range(len(values) - window_size)
                    ]
                    var_trend, _ = np.polyfit(
                        np.arange(len(variances)),
                        variances,
                        1
                    )
                    features.append(var_trend)
                else:
                    features.append(0)
                
                # Recent statistics
                features.extend([
                    np.mean(values[-100:]) if len(values) >= 100 else np.mean(values),
                    np.std(values[-100:]) if len(values) >= 100 else np.std(values),
                    np.max(values[-100:]) if len(values) >= 100 else np.max(values)
                ])
        
        # Operating hours estimate
        hours_operated = len(historical_data) / 3600  # Assuming 1 reading per second
        features.append(hours_operated)
        
        return np.array(features)
    
    async def _generate_maintenance_schedule(
        self,
        predictions: List[Dict[str, Any]]
    ):
        """
        Generate maintenance schedule from predictions
        
        Steps:
        1. Sort machines by priority
        2. Group by maintenance date
        3. Optimize scheduling to avoid conflicts
        4. Store schedule in Firestore
        
        Args:
            predictions: List of RUL predictions
        """
        # Sort by priority
        priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
        sorted_predictions = sorted(
            predictions,
            key=lambda x: (
                priority_order.get(x["priority"], 99),
                x["rul_hours"]
            )
        )
        
        # Create schedule
        schedule = {
            "generated_at": datetime.utcnow().isoformat(),
            "machines": sorted_predictions
        }
        
        # Store in Firestore
        await self.firestore.update_maintenance_schedule(schedule)
        
        logger.info("Maintenance schedule updated")