"""
Cloud Storage Service - UPDATED
Added image storage capability for visualizations
"""

from google.cloud import storage
import logging
from typing import Any
import json
from datetime import datetime

from backend.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service for Cloud Storage operations"""
    
    def __init__(self):
        self.client = storage.Client(project=settings.project_id)
        self.bucket_name = settings.storage_bucket
        self.bucket = self.client.bucket(self.bucket_name)
        
        logger.info(f"StorageService initialized: bucket={self.bucket_name}")
    
    async def store_report(self, filename: str, content: str) -> str:
        """Store analysis report in Cloud Storage"""
        try:
            blob_name = f"{settings.storage_reports_path}{filename}"
            blob = self.bucket.blob(blob_name)
            
            # Determine content type
            if filename.endswith('.html'):
                content_type = 'text/html'
            else:
                content_type = 'application/json'
            
            blob.upload_from_string(content, content_type=content_type)
            blob.make_public()
            
            url = blob.public_url
            logger.info(f"Stored report: {filename}")
            return url
            
        except Exception as e:
            logger.error(f"Error storing report: {e}")
            raise
    
    async def store_visualization_image(self, filename: str, image_bytes: bytes) -> str:
        """
        Store visualization image in Cloud Storage
        
        Args:
            filename: Image filename
            image_bytes: Image data as bytes
            
        Returns:
            Public URL to stored image
        """
        try:
            blob_name = f"{settings.storage_visualizations_path}{filename}"
            blob = self.bucket.blob(blob_name)
            
            blob.upload_from_string(image_bytes, content_type='image/png')
            blob.make_public()
            
            url = blob.public_url
            logger.info(f"Stored visualization image: {filename}")
            return url
            
        except Exception as e:
            logger.error(f"Error storing visualization image: {e}")
            raise
    
    async def store_visualization(self, viz_type: str, data: Any) -> str:
        """Store visualization data (legacy method)"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{viz_type}_{timestamp}.json"
            blob_name = f"{settings.storage_visualizations_path}{filename}"
            
            blob = self.bucket.blob(blob_name)
            content = json.dumps(data, indent=2, default=str)
            blob.upload_from_string(content, content_type='application/json')
            blob.make_public()
            
            url = blob.public_url
            logger.info(f"Stored visualization: {filename}")
            return url
            
        except Exception as e:
            logger.error(f"Error storing visualization: {e}")
            raise
    
    async def get_report(self, filename: str) -> str:
        """Retrieve report from Cloud Storage"""
        try:
            blob_name = f"{settings.storage_reports_path}{filename}"
            blob = self.bucket.blob(blob_name)
            
            content = blob.download_as_string()
            logger.debug(f"Retrieved report: {filename}")
            return content.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error retrieving report: {e}")
            raise