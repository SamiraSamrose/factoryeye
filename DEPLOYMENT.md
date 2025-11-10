# Deployment Guide

Complete guide for deploying FactoryEye to Google Cloud Run.

## Prerequisites

- Google Cloud Platform account
- gcloud CLI installed and authenticated
- Docker installed locally
- Terraform 1.5+ (optional, for infrastructure as code)
- Billing enabled on GCP project

## Deployment Steps

### 1. Project Setup
```bash
# Set project ID
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"

# Configure gcloud
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION
```

### 2. Run Setup Script
```bash
# Make scripts executable
chmod +x deployment/scripts/setup_gcp.sh
chmod +x deployment/scripts/deploy.sh

# Run GCP setup
./deployment/scripts/setup_gcp.sh
```

This creates:
- Pub/Sub topic and subscription
- BigQuery dataset and tables
- Firestore database
- Cloud Storage bucket
- Service account with permissions

### 3. Generate Credentials
```bash
gcloud iam service-accounts keys create credentials.json \
  --iam-account=factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com
```

### 4. Configure Environment

Create `.env` file:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 5. Deploy Services
```bash
./deployment/scripts/deploy.sh
```

This deploys:
- Sensor Sentinel (Cloud Run Service)
- Predictor AI (Cloud Run Job with Cloud Scheduler)
- Efficiency Analyst (Cloud Run Service with public access)

### 6. Verify Deployment
```bash
# Check service status
gcloud run services list --region=$REGION

# Test API endpoint
ANALYST_URL=$(gcloud run services describe factoryeye-analyst \
  --region=$REGION \
  --format="value(status.url)")

curl $ANALYST_URL/health
```

## GPU Configuration

### Enable L4 GPUs

Ensure L4 GPUs are available in your region:
```bash
gcloud compute accelerator-types list \
  --filter="zone:($REGION)" \
  --format="table(name,zone)"
```

### Update Service for GPU
```bash
gcloud run services update factoryeye-sentinel \
  --region=$REGION \
  --gpu=1 \
  --gpu-type=nvidia-l4
```

## Monitoring

### View Logs
```bash
# Sentinel logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=factoryeye-sentinel" \
  --limit=50 \
  --format=json

# Analyst logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=factoryeye-analyst" \
  --limit=50 \
  --format=json
```

### Create Dashboard
```bash
# Access Cloud Console
gcloud console
```

Navigate to Monitoring > Dashboards and create custom dashboard.

## Scaling Configuration

### Auto-scaling Settings
```bash
# Update Sensor Sentinel
gcloud run services update factoryeye-sentinel \
  --region=$REGION \
  --min-instances=1 \
  --max-instances=10 \
  --concurrency=80

# Update Efficiency Analyst
gcloud run services update factoryeye-analyst \
  --region=$REGION \
  --min-instances=1 \
  --max-instances=5 \
  --concurrency=100
```

### Cost Optimization

- Use min-instances=0 for dev environments
- Enable request-based scaling
- Set appropriate memory/CPU limits
- Use Cloud Scheduler for batch jobs

## Security

### Enable VPC Connector
```bash
# Create VPC connector
gcloud compute networks vpc-access connectors create factoryeye-connector \
  --region=$REGION \
  --range=10.8.0.0/28

# Update services
gcloud run services update factoryeye-sentinel \
  --region=$REGION \
  --vpc-connector=factoryeye-connector \
  --vpc-egress=private-ranges-only
```

### Configure IAM
```bash
# Restrict public access
gcloud run services update factoryeye-analyst \
  --region=$REGION \
  --no-allow-unauthenticated

# Add specific users
gcloud run services add-iam-policy-binding factoryeye-analyst \
  --region=$REGION \
  --member="user:email@example.com" \
  --role="roles/run.invoker"
```

## Troubleshooting

### Service Won't Start

Check logs:
```bash
gcloud logging read "resource.type=cloud_run_revision" \
  --limit=100 \
  --format=json
```

### GPU Not Available

Verify GPU quota:
```bash
gcloud compute project-info describe \
  --project=$PROJECT_ID
```

### Permission Denied

Verify service account permissions:
```bash
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:factoryeye-sa@$PROJECT_ID.iam.gserviceaccount.com"
```

## Rollback
```bash
# List revisions
gcloud run revisions list \
  --service=factoryeye-analyst \
  --region=$REGION

# Rollback to previous revision
gcloud run services update-traffic factoryeye-analyst \
  --region=$REGION \
  --to-revisions=REVISION_NAME=100
```

## Cleanup
```bash
# Delete services
gcloud run services delete factoryeye-sentinel --region=$REGION --quiet
gcloud run services delete factoryeye-analyst --region=$REGION --quiet
gcloud run jobs delete factoryeye-predictor --region=$REGION --quiet

# Delete scheduler
gcloud scheduler jobs delete factoryeye-predictor-schedule \
  --location=$REGION --quiet

# Delete other resources
gsutil -m rm -r gs://factoryeye-reports-$PROJECT_ID
bq rm -r -f factoryeye
gcloud pubsub subscriptions delete sensor-data-sub
gcloud pubsub topics delete iot-sensor-data
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Deploy FactoryEye

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}
      - name: Deploy
        run: ./deployment/scripts/deploy.sh
```