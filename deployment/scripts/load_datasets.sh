#!/bin/bash

# Load All Public Datasets into BigQuery
# Executes data loading pipeline

set -e

echo "=========================================="
echo "FactoryEye Dataset Loading"
echo "=========================================="

if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID environment variable is not set"
    exit 1
fi

echo "Project ID: $PROJECT_ID"
echo "Loading public datasets..."

# Run Python dataset loader
python3 << 'END_PYTHON'
import asyncio
import sys
sys.path.insert(0, '.')

from backend.utils.data_loader import DatasetLoader
from backend.config import settings

async def main():
    loader = DatasetLoader()
    await loader.load_all_datasets()
    print("All datasets loaded successfully!")

asyncio.run(main())
END_PYTHON

echo ""
echo "=========================================="
echo "Dataset Loading Complete!"
echo "=========================================="
echo ""
echo "Datasets loaded:"
echo "1. NASA Turbofan Engine Degradation"
echo "2. UCI Hydraulic System Condition"
echo "3. SECOM Manufacturing Data"
echo "4. Kaggle IoT Sensor Logs"
echo ""
echo "Data available in BigQuery dataset: ${PROJECT_ID}.factoryeye"