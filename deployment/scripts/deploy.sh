#!/bin/bash

# FactoryEye Deployment Script
# Deploys all services to Google Cloud Run

set -e

echo "=========================================="
echo "FactoryEye Deployment"
echo "=========================================="

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID environment variable is not set"
    exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Region: ${REGION:-europe-west1}"

REGION=${REGION:-europe-west1}
SERVICE_ACCOUNT="factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com"

# Step 1: Build Docker images
echo ""
echo "Step 1: Building Docker images..."

echo "Building Sensor Sentinel image..."
docker build -f deployment/cloud_run/Dockerfile.sentinel \
    -t gcr.io/$PROJECT_ID/factoryeye-sentinel:latest .

echo "Building Predictor AI image..."
docker build -f deployment/cloud_run/Dockerfile.predictor \
    -t gcr.io/$PROJECT_ID/factoryeye-predictor:latest .

echo "Building Efficiency Analyst image..."
docker build -f deployment/cloud_run/Dockerfile.analyst \
    -t gcr.io/$PROJECT_ID/factoryeye-analyst:latest .

echo "Docker images built successfully"

# Step 2: Push images to Container Registry
echo ""
echo "Step 2: Pushing images to GCR..."

docker push gcr.io/$PROJECT_ID/factoryeye-sentinel:latest
docker push gcr.io/$PROJECT_ID/factoryeye-predictor:latest
docker push gcr.io/$PROJECT_ID/factoryeye-analyst:latest

echo "Images pushed successfully"

# Step 3: Deploy Sensor Sentinel (Worker Pool)
echo ""
echo "Step 3: Deploying Sensor Sentinel..."

gcloud run deploy factoryeye-sentinel \
    --image=gcr.io/$PROJECT_ID/factoryeye-sentinel:latest \
    --platform=managed \
    --region=$REGION \
    --service-account=$SERVICE_ACCOUNT \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=10 \
    --set-env-vars="PROJECT_ID=$PROJECT_ID,REGION=$REGION" \
    --no-allow-unauthenticated \
    --project=$PROJECT_ID

echo "Sensor Sentinel deployed"

# Step 4: Deploy Predictor AI (Cloud Run Job)
echo ""
echo "Step 4: Deploying Predictor AI as Cloud Run Job..."

gcloud run jobs create factoryeye-predictor \
    --image=gcr.io/$PROJECT_ID/factoryeye-predictor:latest \
    --region=$REGION \
    --service-account=$SERVICE_ACCOUNT \
    --memory=8Gi \
    --cpu=4 \
    --task-timeout=3600 \
    --set-env-vars="PROJECT_ID=$PROJECT_ID,REGION=$REGION" \
    --project=$PROJECT_ID || echo "Job already exists, updating..."

gcloud run jobs update factoryeye-predictor \
    --image=gcr.io/$PROJECT_ID/factoryeye-predictor:latest \
    --region=$REGION \
    --project=$PROJECT_ID || echo "Job created successfully"

# Create Cloud Scheduler job for hourly execution
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http factoryeye-predictor-schedule \
    --location=$REGION \
    --schedule="0 * * * *" \
    --uri="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/factoryeye-predictor:run" \
    --http-method=POST \
    --oauth-service-account-email=$SERVICE_ACCOUNT \
    --project=$PROJECT_ID || echo "Scheduler job already exists"

echo "Predictor AI deployed and scheduled"

# Step 5: Deploy Efficiency Analyst (Cloud Run Service)
echo ""
echo "Step 5: Deploying Efficiency Analyst..."

gcloud run deploy factoryeye-analyst \
    --image=gcr.io/$PROJECT_ID/factoryeye-analyst:latest \
    --platform=managed \
    --region=$REGION \
    --service-account=$SERVICE_ACCOUNT \
    --memory=4Gi \
    --cpu=2 \
    --min-instances=1 \
    --max-instances=5 \
    --set-env-vars="PROJECT_ID=$PROJECT_ID,REGION=$REGION" \
    --allow-unauthenticated \
    --project=$PROJECT_ID

ANALYST_URL=$(gcloud run services describe factoryeye-analyst \
    --region=$REGION \
    --format="value(status.url)" \
    --project=$PROJECT_ID)

echo "Efficiency Analyst deployed at: $ANALYST_URL"

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Services deployed:"
echo "1. Sensor Sentinel: factoryeye-sentinel"
echo "2. Predictor AI: factoryeye-predictor (scheduled hourly)"
echo "3. Efficiency Analyst: $ANALYST_URL"
echo ""
echo "Access the dashboard at: $ANALYST_URL"