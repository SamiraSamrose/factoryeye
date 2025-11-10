# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Google Cloud Platform account
- gcloud CLI installed and configured
- Docker (for local testing)
- Terraform 1.5+ (for infrastructure deployment)

## Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/factoryeye.git
cd factoryeye
```

## Step 2: Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Step 3: Configure Google Cloud
```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable firestore.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Create service account
gcloud iam service-accounts create factoryeye-sa \
    --display-name="FactoryEye Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.editor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/datastore.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com
```

## Step 4: Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
```
PROJECT_ID=your-gcp-project-id
REGION=europe-west1
PUBSUB_TOPIC=iot-sensor-data
PUBSUB_SUBSCRIPTION=sensor-data-sub
BIGQUERY_DATASET=factoryeye
FIRESTORE_COLLECTION=sensor_configs
STORAGE_BUCKET=factoryeye-reports
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

## Step 5: Initialize Cloud Resources
```bash
# Run setup script
chmod +x deployment/scripts/setup_gcp.sh
./deployment/scripts/setup_gcp.sh
```

This script creates:
- Pub/Sub topics and subscriptions
- BigQuery dataset and tables
- Firestore database
- Cloud Storage bucket

## Step 6: Local Development
```bash
# Start backend API
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend (optional)
python -m http.server 8080 --directory frontend
```

Access dashboard at http://localhost:8080

## Step 7: Deploy to Cloud Run
```bash
# Deploy all services
chmod +x deployment/scripts/deploy.sh
./deployment/scripts/deploy.sh
```

This deploys:
- Sensor Sentinel (Worker Pool)
- Predictor AI (Cloud Run Job)
- Efficiency Analyst (Cloud Run Service)
- Frontend Dashboard

## Step 8: Verify Deployment
```bash
# Check service status
gcloud run services list --region=europe-west1

# Test API endpoint
curl https://factoryeye-analyst-xxxxx.europe-west1.run.app/health
```

## Troubleshooting

### GPU Not Available
Ensure L4 GPUs are available in europe-west1:
```bash
gcloud compute accelerator-types list --filter="zone:europe-west1"
```

### Permission Denied
Verify service account has all required roles:
```bash
gcloud projects get-iam-policy $PROJECT_ID
```

### Pub/Sub Connection Issues
Check subscription exists and has messages:
```bash
gcloud pubsub subscriptions describe sensor-data-sub
```

## Next Steps

- Review [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Follow [DEMO_GUIDE.md](docs/DEMO_GUIDE.md) to test the system
- Check [API.md](docs/API.md) for integration details