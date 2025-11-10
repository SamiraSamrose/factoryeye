# FactoryEye API Documentation

## Base URL
```
https://factoryeye-analyst-xxxxx.europe-west1.run.app
```

## Authentication

API endpoints use Cloud IAM authentication. For public access to the dashboard, no authentication is required for GET endpoints.

## Endpoints

### Health Check
```
GET /health
```

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "service": "factoryeye-api",
  "version": "1.0.0"
}
```

---

### List Sensors
```
GET /api/v1/sensors
```

Retrieve all active sensor configurations.

**Response:**
```json
{
  "count": 10,
  "sensors": [
    {
      "sensor_id": "sensor_001",
      "machine_id": "machine_001",
      "active": true,
      "thresholds": {
        "temperature": {"min": 20, "max": 85},
        "vibration": {"min": 0, "max": 50},
        "pressure": {"min": 1, "max": 10},
        "rpm": {"min": 0, "max": 5000}
      }
    }
  ]
}
```

---

### Get Sensor Details
```
GET /api/v1/sensors/{sensor_id}
```

Get configuration for a specific sensor.

**Parameters:**
- `sensor_id` (path) - Sensor identifier

**Response:**
```json
{
  "sensor_id": "sensor_001",
  "machine_id": "machine_001",
  "active": true,
  "thresholds": {...}
}
```

---

### List Anomalies
```
GET /api/v1/anomalies
```

Query recent anomalies.

**Query Parameters:**
- `hours` (optional, default: 24) - Time range in hours (1-168)
- `machine_id` (optional) - Filter by machine ID

**Response:**
```json
{
  "count": 5,
  "time_range_hours": 24,
  "machine_id": null,
  "anomalies": [
    {
      "sensor_id": "sensor_001",
      "timestamp": "2024-01-15T10:25:00Z",
      "anomaly_score": 0.92,
      "severity": "high",
      "metrics": {
        "temperature": 88.5,
        "vibration": 45.2,
        "pressure": 8.1,
        "rpm": 4800
      }
    }
  ]
}
```

---

### List Predictions
```
GET /api/v1/predictions
```

Get RUL predictions for machines.

**Query Parameters:**
- `machine_id` (optional) - Filter by machine ID

**Response:**
```json
{
  "count": 3,
  "machine_id": null,
  "predictions": [
    {
      "machine_id": "machine_001",
      "rul_hours": 168.5,
      "confidence": 0.87,
      "priority": "medium",
      "failure_modes": ["bearing_degradation"],
      "recommended_maintenance_date": "2024-01-22T10:00:00Z",
      "predicted_at": "2024-01-15T09:00:00Z"
    }
  ]
}
```

---

### Request Efficiency Analysis
```
POST /api/v1/analyze
```

Trigger efficiency analysis with Efficiency Analyst agent.

**Query Parameters:**
- `machine_id` (optional) - Analyze specific machine
- `time_range_hours` (optional, default: 24) - Analysis time range (1-168)

**Response:**
```json
{
  "machine_id": "machine_001",
  "time_range_hours": 24,
  "analyzed_at": "2024-01-15T10:30:00Z",
  "kpis": {
    "uptime_percent": 98.5,
    "anomaly_rate_percent": 2.1,
    "efficiency_score": 94.3,
    "avg_temperature": 65.2,
    "avg_vibration": 18.5,
    "avg_pressure": 5.8,
    "avg_rpm": 3200,
    "total_anomalies": 5,
    "estimated_cost_savings_usd": 12500.00
  },
  "insights": "Overall performance is strong with 98.5% uptime...",
  "visualizations": {
    "time_series": "https://storage.googleapis.com/...",
    "heatmap": "https://storage.googleapis.com/..."
  },
  "report_url": "https://storage.googleapis.com/.../report.json"
}
```

---

### Dashboard Statistics
```
GET /api/v1/dashboard/stats
```

Get aggregated statistics for dashboard overview.

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "stats": {
    "total_machines": 5,
    "total_sensors": 20,
    "total_readings": 86400,
    "total_anomalies": 12,
    "avg_anomaly_score": 0.89,
    "urgent_maintenance_count": 1
  }
}
```

---

### List Machines
```
GET /api/v1/machines
```

Get all machines with sensor counts.

**Response:**
```json
{
  "count": 5,
  "machines": [
    {
      "machine_id": "machine_001",
      "active": true,
      "sensor_count": 4
    }
  ]
}
```

---

## WebSocket

### Real-Time Updates
```
ws://factoryeye-analyst-xxxxx.europe-west1.run.app/ws
```

Connect to WebSocket for real-time anomaly and metric updates.

**Message Format:**
```json
{
  "type": "anomaly",
  "data": {
    "sensor_id": "sensor_001",
    "timestamp": "2024-01-15T10:30:00Z",
    "is_anomaly": true,
    "anomaly_score": 0.92,
    "metrics": {...}
  }
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

- `200` - Success
- `400` - Bad Request
- `404` - Not Found
- `500` - Internal Server Error

**Error Format:**
```json
{
  "detail": "Error message description"
}
```

---

## Rate Limits

- API requests: 1000 per hour per IP
- WebSocket connections: 10 concurrent per IP

---

## Examples

### Python
```python
import requests

# List anomalies
response = requests.get(
    'https://factoryeye-analyst-xxxxx.europe-west1.run.app/api/v1/anomalies',
    params={'hours': 24}
)
anomalies = response.json()

# Request analysis
response = requests.post(
    'https://factoryeye-analyst-xxxxx.europe-west1.run.app/api/v1/analyze',
    params={'time_range_hours': 24}
)
analysis = response.json()
```

### JavaScript
```javascript
// Fetch predictions
fetch('https://factoryeye-analyst-xxxxx.europe-west1.run.app/api/v1/predictions')
  .then(response => response.json())
  .then(data => console.log(data));

// WebSocket connection
const ws = new WebSocket('wss://factoryeye-analyst-xxxxx.europe-west1.run.app/ws');
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Received:', message);
};
```