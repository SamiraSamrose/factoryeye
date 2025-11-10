# FactoryEye – Industrial IoT Anomaly & Efficiency Optimizer

Production-ready multi-agent AI system for real-time industrial IoT monitoring, anomaly detection, and predictive maintenance using Google Cloud Run with GPU acceleration.

## Overview

FactoryEye leverages a multi-agent architecture to:
- Detect anomalies in real-time sensor data (Agent 1: Sensor Sentinel)
- Predict equipment failures and maintenance needs (Agent 2: Predictor AI)
- Analyze efficiency and generate insights (Agent 3: Efficiency Analyst)

## Architecture

The system consists of three AI agents running on Google Cloud Run:
- **Sensor Sentinel**: Real-time anomaly detection using Gemma on L4 GPU
- **Predictor AI**: Batch predictive maintenance using Gemma on L4 GPU
- **Efficiency Analyst**: Insight generation using Gemini, visualizations with Imagen

Data flows through MQTT to Pub/Sub, processed by agents, stored in BigQuery, with configurations in Firestore and reports in Cloud Storage.

## Links

- **Live Site Demo**: https://samirasamrose.github.io/factoryeye/
- **Source Code**: https://github.com/SamiraSamrose/factoryeye
- **Video Demo**: https://youtu.be/zXTwUWvDwbM

## Technology Stack

- **Backend**: Python 3.11, FastAPI, asyncio, uvicorn
- **Frontend**: HTML5, CSS3, JavaScript ES6, Vue.js 3, Bootstrap 5, Chart.js 4.4, Plotly.js 2.27
- **Cloud Infrastructure**: Google Cloud Run (Services, Jobs, Worker Pools), Cloud Pub/Sub, BigQuery, Firestore, Cloud Storage
- **AI/ML Models**: Gemma-7B (anomaly detection, RUL prediction), Gemini 1.5 Pro (natural language insights), Imagen 3.0 (visualization generation), Veo (video generation)
- **GPU**: NVIDIA L4 (europe-west1 region)
- **ML Libraries**: PyTorch 2.1, Transformers 4.35, scikit-learn 1.3, NumPy 1.24, Pandas 2.0, SciPy 1.11
- **Visualization**: Matplotlib 3.8, Seaborn 0.13, Plotly 5.18, WebGL rendering
- **Data Processing**: pyarrow 14.0, jsonschema 4.20
- **Google Cloud SDKs**: google-cloud-pubsub 2.18, google-cloud-bigquery 3.13, google-cloud-firestore 2.13, google-cloud-storage 2.10, google-cloud-aiplatform 1.38, vertexai 1.38
- **API/WebSocket**: websockets 12.0, aiohttp 3.9, python-multipart 0.0.6
- **Deployment**: Docker, Terraform 1.5, bash scripts
- **Testing**: pytest 7.4, pytest-asyncio 0.21, httpx 0.25
- **Development Tools**: black 23.12, flake8 6.1, mypy 1.7

## Data Integrations & Datasets

- **NASA Turbofan Engine Degradation Dataset**: Run-to-failure turbofan engine data with 21 sensor measurements across multiple operational cycles, used for training RUL prediction models
- **UCI Hydraulic System Condition Dataset**: Hydraulic test rig data including pressure, flow, temperature, vibration measurements with labeled degradation states
- **SECOM Manufacturing Dataset**: Semiconductor manufacturing process data with 590 sensor measurements and binary pass/fail labels
- **Kaggle IoT Sensor Logs**: Generic IoT telemetry data with temperature, humidity, pressure, motion detection across multiple sensors
- **Data Processing**: Automated ETL pipeline loads datasets into BigQuery, performs feature engineering (rolling statistics, rate of change, cross-sensor interactions), handles missing values, applies normalization


## Quick Start

See [SETUP.md](SETUP.md) for detailed installation and deployment instructions.

## Features

- Real-time anomaly detection with sub-second latency
- Predictive maintenance with RUL (Remaining Useful Life) forecasting
- Interactive web dashboard with live metrics
- Automated alert system for critical anomalies
- Cost optimization through serverless GPU acceleration
- Comprehensive API for integration


## Documentation

- [Setup Guide](SETUP.md)
- [Architecture Details](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Deployment Guide](DEPLOYMENT.md)

## License

MIT License
