"""
Main FastAPI Application
Provides REST API and WebSocket endpoints for dashboard and integrations
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
import asyncio
from typing import List
from datetime import datetime

from backend.config import settings
from backend.api.routes import router
from backend.api.websocket import WebSocketManager
from backend.services.pubsub_service import PubSubService
from backend.agents.sensor_sentinel import SensorSentinel

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# Initialize services
pubsub_service = PubSubService()
sensor_sentinel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler
    Manages startup and shutdown of background services
    """
    logger.info("Starting FactoryEye application...")
    
    # Initialize Sensor Sentinel agent
    global sensor_sentinel
    sensor_sentinel = SensorSentinel()
    
    # Start background tasks
    asyncio.create_task(process_sensor_stream())
    asyncio.create_task(ws_manager.broadcast_loop())
    
    logger.info("Application started successfully")
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    await ws_manager.disconnect_all()
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="FactoryEye API",
    description="Industrial IoT Anomaly & Efficiency Optimizer",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


async def process_sensor_stream():
    """
    Background task: Process sensor data stream from Pub/Sub
    Runs Sensor Sentinel agent for real-time anomaly detection
    """
    logger.info("Starting sensor stream processing...")
    
    try:
        async for messages in pubsub_service.pull_messages_async():
            for message in messages:
                try:
                    # Process message with Sensor Sentinel
                    result = await sensor_sentinel.process_message(message)
                    
                    # Broadcast anomalies to connected clients
                    if result.get("is_anomaly"):
                        await ws_manager.broadcast({
                            "type": "anomaly",
                            "data": result
                        })
                    
                    # Acknowledge message
                    message.ack()
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    message.nack()
                    
    except Exception as e:
        logger.error(f"Error in sensor stream processing: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main dashboard"""
    with open("frontend/templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve interactive demo interface"""
    with open("frontend/templates/demo.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/analytics", response_class=HTMLResponse)
async def analytics():
    """Serve analytics dashboard"""
    with open("frontend/templates/analytics.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Used by Cloud Run for container health monitoring
    """
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "factoryeye-api",
        "version": "1.0.0"
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates
    Streams anomalies and metrics to connected dashboard clients
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )