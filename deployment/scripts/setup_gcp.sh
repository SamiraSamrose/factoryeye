#!/bin/bash

# FactoryEye GCP Setup Script
# Creates all required Google Cloud resources

set -e

echo "=========================================="
echo "FactoryEye GCP Setup"
echo "=========================================="

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID environment variable is not set"
    echo "Usage: export PROJECT_ID=your-project-id && ./setup_gcp.sh"
    exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Region: ${REGION:-europe-west1}"

REGION=${REGION:-europe-west1}

# Step 1: Enable required APIs
echo ""
echo "Step 1: Enabling required APIs..."
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable pubsub.googleapis.com --project=$PROJECT_ID
gcloud services enable bigquery.googleapis.com --project=$PROJECT_ID
gcloud services enable firestore.googleapis.com --project=$PROJECT_ID
gcloud services enable storage.googleapis.com --project=$PROJECT_ID
gcloud services enable aiplatform.googleapis.com --project=$PROJECT_ID
echo "APIs enabled successfully"

# Step 2: Create Pub/Sub resources
echo ""
echo "Step 2: Creating Pub/Sub topic and subscription..."
gcloud pubsub topics create iot-sensor-data --project=$PROJECT_ID || echo "Topic already exists"
gcloud pubsub subscriptions create sensor-data-sub \
    --topic=iot-sensor-data \
    --ack-deadline=300 \
    --project=$PROJECT_ID || echo "Subscription already exists"
echo "Pub/Sub resources created"

# Step 3: Create BigQuery dataset and tables
echo ""
echo "Step 3: Creating BigQuery dataset and tables..."
bq --location=$REGION mk --dataset $PROJECT_ID:factoryeye || echo "Dataset already exists"

# Create sensor_readings table
bq mk --table $PROJECT_ID:factoryeye.sensor_readings \
    sensor_id:STRING,timestamp:TIMESTAMP,machine_id:STRING,metrics:RECORD \
    || echo "sensor_readings table already exists"

# Create anomalies table
bq mk --table $PROJECT_ID:factoryeye.anomalies \
    sensor_id:STRING,timestamp:TIMESTAMP,anomaly_score:FLOAT,severity:STRING,metrics:RECORD,detected_at:TIMESTAMP \
    || echo "anomalies table already exists"

# Create predictions table
bq mk --table $PROJECT_ID:factoryeye.predictions \
    machine_id:STRING,rul_hours:FLOAT,confidence:FLOAT,priority:STRING,failure_modes:STRING,recommended_maintenance_date:TIMESTAMP,predicted_at:TIMESTAMP \
    || echo "predictions table already exists"

# Create efficiency_metrics table
bq mk --table $PROJECT_ID:factoryeye.efficiency_metrics \
    machine_id:STRING,timestamp:TIMESTAMP,efficiency_score:FLOAT,uptime_percent:FLOAT,anomaly_rate:FLOAT \
    || echo "efficiency_metrics table already exists"

echo "BigQuery resources created"

# Step 4: Create Firestore database
echo ""
echo "Step 4: Creating Firestore database..."
gcloud firestore databases create --location=$REGION --project=$PROJECT_ID || echo "Firestore database already exists"
echo "Firestore database ready"

# Step 5: Create Cloud Storage bucket
echo ""
echo "Step 5: Creating Cloud Storage bucket..."
gsutil mb -l $REGION gs://factoryeye-reports-$PROJECT_ID || echo "Bucket already exists"
gsutil uniformbucketlevelaccess set on gs://factoryeye-reports-$PROJECT_ID
echo "Cloud Storage bucket created"

# Step 6: Create service account
echo ""
echo "Step 6: Creating service account..."
gcloud iam service-accounts create factoryeye-sa \
    --display-name="FactoryEye Service Account" \
    --project=$PROJECT_ID || echo "Service account already exists"


# Grant necessary permissions
echo "Granting IAM permissions..."
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

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Step 7: Load datasets
echo ""
echo "Step 7: Loading public datasets..."
python3 -c "
from backend.utils.data_loader import DatasetLoader
import asyncio

async def load_all():
    loader = DatasetLoader()
    await loader.load_nasa_turbofan()
    await loader.load_uci_hydraulic()
    await loader.load_secom()
    await loader.load_kaggle_iot()

asyncio.run(load_all())
"
echo "Datasets loaded into BigQuery"

echo "Service account configured"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Generate service account key:"
echo "   gcloud iam service-accounts keys create credentials.json \\"
echo "     --iam-account=factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com"
echo ""
echo "2. Update .env file with your configuration"
echo ""
echo "3. Deploy services using: ./deployment/scripts/deploy.sh"