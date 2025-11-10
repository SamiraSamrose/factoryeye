"""
Pub/Sub Service
Handles message publishing and subscription for IoT sensor data
Manages connection to Google Cloud Pub/Sub
"""

from google.cloud import pubsub_v1
from google.api_core import retry
import logging
from typing import List, AsyncIterator
import asyncio
import json

from backend.config import settings

logger = logging.getLogger(__name__)


class PubSubService:
    """
    Service for Pub/Sub operations
    Provides methods for publishing and consuming sensor messages
    """
    
    def __init__(self):
        self.project_id = settings.project_id
        self.topic_name = settings.pubsub_topic
        self.subscription_name = settings.pubsub_subscription
        
        # Initialize publisher and subscriber clients
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Build topic and subscription paths
        self.topic_path = self.publisher.topic_path(
            self.project_id,
            self.topic_name
        )
        self.subscription_path = self.subscriber.subscription_path(
            self.project_id,
            self.subscription_name
        )
        
        logger.info(f"PubSubService initialized: topic={self.topic_path}")
    
    async def publish_message(self, data: dict) -> str:
        """
        Publish message to Pub/Sub topic
        
        Steps:
        1. Serialize data to JSON
        2. Encode to bytes
        3. Publish to topic
        4. Return message ID
        
        Args:
            data: Message data dictionary
            
        Returns:
            Published message ID
        """
        try:
            # Serialize data
            message_json = json.dumps(data)
            message_bytes = message_json.encode('utf-8')
            
            # Publish message
            future = self.publisher.publish(
                self.topic_path,
                message_bytes
            )
            
            # Wait for publish to complete
            message_id = await asyncio.get_event_loop().run_in_executor(
                None,
                future.result
            )
            
            logger.debug(f"Published message: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            raise
    
    async def pull_messages_async(
        self,
        max_messages: int = None
    ) -> AsyncIterator[List]:
        """
        Pull messages from subscription asynchronously
        
        Steps:
        1. Pull batch of messages from subscription
        2. Yield messages for processing
        3. Continue pulling in loop
        
        Args:
            max_messages: Maximum messages per pull
            
        Yields:
            List of messages from subscription
        """
        if max_messages is None:
            max_messages = settings.pubsub_max_messages
        
        logger.info(f"Starting message pull loop: max_messages={max_messages}")
        
        while True:
            try:
                # Pull messages
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.subscriber.pull(
                        request={
                            "subscription": self.subscription_path,
                            "max_messages": max_messages
                        },
                        timeout=30.0
                    )
                )
                
                if response.received_messages:
                    logger.debug(
                        f"Pulled {len(response.received_messages)} messages"
                    )
                    yield response.received_messages
                else:
                    # No messages, wait before next pull
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error pulling messages: {e}")
                await asyncio.sleep(5)  # Wait before retry
    
    def acknowledge_messages(self, ack_ids: List[str]):
        """
        Acknowledge processed messages
        
        Steps:
        1. Batch acknowledge messages by IDs
        2. Remove from subscription queue
        
        Args:
            ack_ids: List of message acknowledgment IDs
        """
        try:
            self.subscriber.acknowledge(
                request={
                    "subscription": self.subscription_path,
                    "ack_ids": ack_ids
                }
            )
            logger.debug(f"Acknowledged {len(ack_ids)} messages")
            
        except Exception as e:
            logger.error(f"Error acknowledging messages: {e}")