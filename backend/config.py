"""
Configuration Management
Loads and validates environment variables and application settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Google Cloud Configuration
    project_id: str = Field(..., env="PROJECT_ID")
    region: str = Field(default="europe-west1", env="REGION")
    credentials_path: str = Field(
        default="./credentials.json",
        env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    
    # Pub/Sub Configuration
    pubsub_topic: str = Field(default="iot-sensor-data", env="PUBSUB_TOPIC")
    pubsub_subscription: str = Field(
        default="sensor-data-sub",
        env="PUBSUB_SUBSCRIPTION"
    )
    pubsub_max_messages: int = Field(default=100, env="PUBSUB_MAX_MESSAGES")
    pubsub_ack_deadline: int = Field(default=300, env="PUBSUB_ACK_DEADLINE")
    
    # BigQuery Configuration
    bigquery_dataset: str = Field(default="factoryeye", env="BIGQUERY_DATASET")
    bigquery_table_readings: str = Field(
        default="sensor_readings",
        env="BIGQUERY_TABLE_READINGS"
    )
    bigquery_table_anomalies: str = Field(
        default="anomalies",
        env="BIGQUERY_TABLE_ANOMALIES"
    )
    bigquery_table_predictions: str = Field(
        default="predictions",
        env="BIGQUERY_TABLE_PREDICTIONS"
    )
    bigquery_table_metrics: str = Field(
        default="efficiency_metrics",
        env="BIGQUERY_TABLE_METRICS"
    )
    
    # Firestore Configuration
    firestore_collection: str = Field(
        default="sensor_configs",
        env="FIRESTORE_COLLECTION"
    )
    firestore_alerts_collection: str = Field(
        default="alert_rules",
        env="FIRESTORE_ALERTS_COLLECTION"
    )
    
    # Cloud Storage Configuration
    storage_bucket: str = Field(
        default="factoryeye-reports",
        env="STORAGE_BUCKET"
    )
    storage_reports_path: str = Field(
        default="reports/",
        env="STORAGE_REPORTS_PATH"
    )
    storage_visualizations_path: str = Field(
        default="visualizations/",
        env="STORAGE_VISUALIZATIONS_PATH"
    )
    
    # AI Model Configuration
    gemma_model_id: str = Field(default="gemma-7b-it", env="GEMMA_MODEL_ID")
    gemini_model_id: str = Field(
        default="gemini-1.5-pro",
        env="GEMINI_MODEL_ID"
    )
    imagen_model_id: str = Field(
        default="imagen-3.0-generate-001",
        env="IMAGEN_MODEL_ID"
    )
    
    # Application Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    workers: int = Field(default=4, env="WORKERS")
    
    # Agent Configuration
    anomaly_threshold: float = Field(default=0.85, env="ANOMALY_THRESHOLD")
    prediction_window_hours: int = Field(
        default=168,
        env="PREDICTION_WINDOW_HOURS"
    )
    analysis_batch_size: int = Field(
        default=1000,
        env="ANALYSIS_BATCH_SIZE"
    )
    
    # Frontend Configuration
    ws_reconnect_interval: int = Field(
        default=5000,
        env="WS_RECONNECT_INTERVAL"
    )
    chart_update_interval: int = Field(
        default=1000,
        env="CHART_UPDATE_INTERVAL"
    )
    dashboard_refresh_rate: int = Field(
        default=5000,
        env="DASHBOARD_REFRESH_RATE"
    )
    
    @validator("credentials_path")
    def validate_credentials(cls, v):
        """Validate that credentials file exists"""
        if not os.path.exists(v):
            raise ValueError(f"Credentials file not found: {v}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()