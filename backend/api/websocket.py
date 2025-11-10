"""
WebSocket Manager
Handles real-time communication with dashboard clients
Broadcasts anomalies and metrics to connected clients
"""

from fastapi import WebSocket
import logging
import asyncio
import json
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time updates
    Broadcasts events to all connected clients
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("WebSocketManager initialized")
    
    async def connect(self, websocket: WebSocket):
        """
        Accept new WebSocket connection
        
        Steps:
        1. Accept connection
        2. Add to active connections list
        3. Send welcome message
        
        Args:
            websocket: WebSocket connection to add
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to FactoryEye real-time stream"
        })
        
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection
        
        Steps:
        1. Remove from active connections
        2. Log disconnection
        
        Args:
            websocket: WebSocket connection to remove
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def disconnect_all(self):
        """
        Disconnect all active WebSocket connections
        
        Called during application shutdown
        """
        for connection in self.active_connections:
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.active_connections.clear()
        logger.info("All WebSocket connections closed")
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected clients
        
        Steps:
        1. Serialize message to JSON
        2. Send to all active connections
        3. Handle disconnected clients
        
        Args:
            message: Message dictionary to broadcast
        """
        await self.broadcast_queue.put(message)
    
    async def broadcast_loop(self):
        """
        Background task to process broadcast queue
        
        Continuously pulls messages from queue and sends to clients
        """
        logger.info("Starting WebSocket broadcast loop")
        
        while True:
            try:
                # Get message from queue
                message = await self.broadcast_queue.get()
                
                # Send to all connections
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending to WebSocket: {e}")
                        disconnected.append(connection)
                
                # Remove disconnected clients
                for connection in disconnected:
                    self.disconnect(connection)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1)