# FactoryEye Architecture

## System Overview

FactoryEye implements a multi-agent architecture for industrial IoT monitoring with three specialized AI agents running on Google Cloud Run.

## Data Flow
```
IoT Sensors → MQTT Broker → Cloud Pub/Sub → Agent Processing → Storage → Dashboard
```

### 1. Data Ingestion Layer

**Components:**
- MQTT Broker: Receives sensor telemetry
- Cloud Pub/Sub: Message queue for reliable delivery
- Input validation and preprocessing

**Data Schema:**
```json
{
  "sensor_id": "string",
  "timestamp": "ISO8601",
  "machine_id": "string",
  "metrics": {
    "temperature": "float",
    "vibration": "float",
    "pressure": "float",
    "rpm": "float"
  }
}
```

### 2. Agent Layer

#### Agent 1: Sensor Sentinel (Real-time Processing)
**Purpose:** Detect anomalies in streaming sensor data

**Implementation:**
- Cloud Run Worker Pool with L4 GPU
- Gemma model for anomaly detection
- Sliding window analysis (30-second windows)
- Statistical and ML-based detection

**Processing:**
1. Receives messages from Pub/Sub
2. Extracts features from sensor readings
3. Runs Gemma inference on GPU
4. Flags anomalies with confidence scores
5. Writes results to BigQuery
6. Triggers alerts for critical anomalies

#### Agent 2: Predictor AI (Batch Processing)
**Purpose:** Predict equipment failures and maintenance needs

**Implementation:**
- Cloud Run Job with L4 GPU
- Gemma model for RUL prediction
- Scheduled execution (hourly)
- Historical data analysis

**Processing:**
1. Queries historical data from BigQuery
2. Aggregates sensor patterns
3. Trains/updates prediction model
4. Generates RUL forecasts
5. Stores predictions in BigQuery
6. Creates maintenance schedules

#### Agent 3: Efficiency Analyst (Analysis & Visualization)
**Purpose:** Generate insights and visualizations

**Implementation:**
- Cloud Run Service
- Gemini for natural language insights
- Imagen for chart generation
- REST API for dashboard

**Processing:**
1. Receives analysis requests via API
2. Queries aggregated data from BigQuery
3. Generates insights with Gemini
4. Creates visualizations with Imagen
5. Stores reports in Cloud Storage
6. Serves results to frontend

### 3. Storage Layer

**BigQuery Tables:**
- `sensor_readings`: Raw telemetry data
- `anomalies`: Detected anomalies with metadata
- `predictions`: RUL forecasts and maintenance schedules
- `efficiency_metrics`: Aggregated performance data

**Firestore Collections:**
- `sensor_configs`: Sensor metadata and thresholds
- `alert_rules`: Alert configuration
- `user_preferences`: Dashboard settings

**Cloud Storage Buckets:**
- `reports/`: Generated analysis reports
- `visualizations/`: Imagen-generated charts
- `logs/`: Application logs

### 4. API Layer

**FastAPI Endpoints:**
- `GET /health`: Service health check
- `GET /api/v1/sensors`: List active sensors
- `GET /api/v1/anomalies`: Query anomalies
- `GET /api/v1/predictions`: Get RUL forecasts
- `POST /api/v1/analyze`: Request analysis
- `WebSocket /ws`: Real-time updates

### 5. Frontend Layer

**Dashboard Components:**
- Real-time metrics display
- Anomaly timeline chart
- Equipment heatmap
- RUL prediction graph
- Efficiency trends
- Alert notification panel

**Technologies:**
- HTML5/CSS3 for layout
- Chart.js for visualizations
- WebSocket for live updates
- Responsive design for mobile

## Deployment Architecture
```
europe-west1 Region
├── Cloud Run Services
│   ├── sensor-sentinel (Worker Pool, L4 GPU)
│   ├── predictor-ai (Job, L4 GPU)
│   └── efficiency-analyst (Service)
├── Cloud Pub/Sub
│   └── iot-sensor-data topic
├── BigQuery
│   └── factoryeye dataset
├── Firestore
│   └── (default) database
└── Cloud Storage
    └── factoryeye-reports bucket
```

## Scaling Strategy

**Horizontal Scaling:**
- Sensor Sentinel: Auto-scale 1-10 instances based on Pub/Sub queue depth
- Predictor AI: Manual scaling via job concurrency
- Efficiency Analyst: Auto-scale 1-5 instances based on CPU utilization

**Vertical Scaling:**
- GPU instances for compute-intensive workloads
- CPU-only instances for API services

## Security

- Service account with least-privilege IAM roles
- VPC connector for private networking
- API authentication via Cloud IAM
- Encrypted data at rest and in transit
- Secret management via Secret Manager

## Monitoring

- Cloud Logging for application logs
- Cloud Monitoring for metrics
- Custom dashboards for KPIs
- Alert policies for critical events
- Trace analysis for performance optimization
