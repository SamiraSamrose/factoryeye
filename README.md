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

## Technology Stack

- **Backend**: Python 3.11, FastAPI, asyncio
- **AI Models**: Gemma (anomaly detection), Gemini (insights), Imagen (visualizations)
- **Cloud Infrastructure**: Google Cloud Run, Pub/Sub, BigQuery, Firestore, Cloud Storage
- **Frontend**: HTML5, JavaScript, Chart.js, WebSocket
- **GPU**: NVIDIA L4 (europe-west1 region)

## Links

- **Live Site Demo**: https://samirasamrose.github.io/factoryeye/
- **Source Code**: https://github.com/SamiraSamrose/factoryeye
- **Video Demo**: https://youtu.be/zXTwUWvDwbM

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
