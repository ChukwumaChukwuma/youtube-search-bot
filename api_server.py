from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncio
import uuid
import time
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict
import uvicorn
from contextlib import asynccontextmanager
import os
import psutil
import gc

# Import our bot components
from main import YouTubeSearchBot
from captcha_solver import AdvancedCaptchaSolver
from tls_fingerprinting import TLSFingerprintManager
from ml_behavior import MLBehaviorEngine
from proxy_rotation import ProxyRotationManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bot instance and management
bot_manager = None
auto_scaler = None

class SearchRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=200)
    max_results: int = Field(50, ge=1, le=500)
    session_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = {}

class SearchResponse(BaseModel):
    search_id: str
    status: str
    results: Optional[List[Dict]] = None
    error: Optional[str] = None
    timestamp: str
    duration_ms: Optional[int] = None

class SystemStatus(BaseModel):
    status: str
    active_searches: int
    total_browsers: int
    cpu_usage: float
    memory_usage: float
    uptime_seconds: float
    total_searches_completed: int
    average_search_time_ms: float
    success_rate: float

class BotManager:
    """Manages bot instances and load distribution"""

    def __init__(self):
        self.bots: List[YouTubeSearchBot] = []
        self.search_queue = asyncio.Queue()
        self.results_cache = {}
        self.metrics = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_search_time': 0,
            'start_time': time.time()
        }
        self.active_searches = {}
        self.rate_limiter = RateLimiter()

    async def initialize(self, initial_bots: int = 5):
        """Initialize bot manager with initial bot pool"""
        logger.info(f"Initializing BotManager with {initial_bots} bots")

        # Create initial bot instances
        for i in range(initial_bots):
            bot = YouTubeSearchBot(max_browsers=2)
            await bot.initialize()
            self.bots.append(bot)

        # Start background workers
        for i in range(initial_bots * 2):  # 2 workers per bot
            asyncio.create_task(self._search_worker(f"worker-{i}"))

        logger.info("BotManager initialized successfully")

    async def submit_search(self, request: SearchRequest) -> str:
        """Submit a search request to the queue"""
        # Check rate limits
        client_id = request.session_id or "anonymous"
        if not await self.rate_limiter.check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Generate search ID
        search_id = str(uuid.uuid4())

        # Add to queue
        search_task = {
            'id': search_id,
            'request': request,
            'timestamp': datetime.now(),
            'status': 'queued'
        }

        self.active_searches[search_id] = search_task
        await self.search_queue.put(search_task)

        return search_id

    async def get_search_status(self, search_id: str) -> Optional[Dict]:
        """Get status of a search"""
        if search_id in self.results_cache:
            return self.results_cache[search_id]
        elif search_id in self.active_searches:
            task = self.active_searches[search_id]
            return {
                'search_id': search_id,
                'status': task['status'],
                'timestamp': task['timestamp'].isoformat()
            }
        else:
            return None

    async def _search_worker(self, worker_id: str):
        """Background worker to process search requests"""
        logger.info(f"Search worker {worker_id} started")

        while True:
            try:
                # Get search task from queue
                task = await self.search_queue.get()

                # Update status
                task['status'] = 'processing'
                search_id = task['id']
                request = task['request']

                # Select bot with least load
                bot = self._select_bot()

                # Perform search
                start_time = time.time()

                try:
                    results = await bot.search(
                        keyword=request.keyword,
                        max_results=request.max_results
                    )

                    duration_ms = int((time.time() - start_time) * 1000)

                    # Store results
                    self.results_cache[search_id] = {
                        'search_id': search_id,
                        'status': 'completed',
                        'results': results,
                        'timestamp': datetime.now().isoformat(),
                        'duration_ms': duration_ms
                    }

                    # Update metrics
                    self.metrics['total_searches'] += 1
                    self.metrics['successful_searches'] += 1
                    self.metrics['total_search_time'] += duration_ms

                except Exception as e:
                    logger.error(f"Search error: {e}")

                    self.results_cache[search_id] = {
                        'search_id': search_id,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }

                    self.metrics['total_searches'] += 1
                    self.metrics['failed_searches'] += 1

                # Clean up
                if search_id in self.active_searches:
                    del self.active_searches[search_id]

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    def _select_bot(self) -> YouTubeSearchBot:
        """Select bot with least current load"""
        # Simple round-robin for now
        # In production, would track actual bot load
        return min(self.bots, key=lambda b: b.active_searches)

    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        uptime = time.time() - self.metrics['start_time']
        total_searches = self.metrics['total_searches']

        avg_search_time = 0
        success_rate = 0

        if total_searches > 0:
            avg_search_time = self.metrics['total_search_time'] / total_searches
            success_rate = self.metrics['successful_searches'] / total_searches

        return SystemStatus(
            status='operational',
            active_searches=len(self.active_searches),
            total_browsers=sum(len(bot.browser_pool) for bot in self.bots),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            uptime_seconds=uptime,
            total_searches_completed=total_searches,
            average_search_time_ms=avg_search_time,
            success_rate=success_rate
        )

    async def scale_up(self, additional_bots: int):
        """Scale up by adding more bots"""
        logger.info(f"Scaling up: adding {additional_bots} bots")

        for i in range(additional_bots):
            bot = YouTubeSearchBot(max_browsers=2)
            await bot.initialize()
            self.bots.append(bot)

            # Add workers for new bot
            for j in range(2):
                worker_id = f"worker-scaled-{len(self.bots)}-{j}"
                asyncio.create_task(self._search_worker(worker_id))

    async def scale_down(self, remove_bots: int):
        """Scale down by removing bots"""
        logger.info(f"Scaling down: removing {remove_bots} bots")

        for i in range(min(remove_bots, len(self.bots) - 1)):
            if len(self.bots) > 1:  # Keep at least one bot
                bot = self.bots.pop()
                await bot.cleanup()

    async def cleanup(self):
        """Clean up all resources"""
        for bot in self.bots:
            await bot.cleanup()
        self.bots.clear()


class RateLimiter:
    """Rate limiting for API requests"""

    def __init__(self):
        self.requests = defaultdict(list)
        self.limits = {
            'requests_per_minute': 60,
            'requests_per_hour': 1000,
            'requests_per_day': 10000
        }

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        now = datetime.now()

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < timedelta(days=1)
        ]

        # Check limits
        minute_requests = sum(
            1 for req_time in self.requests[client_id]
            if now - req_time < timedelta(minutes=1)
        )

        if minute_requests >= self.limits['requests_per_minute']:
            return False

        # Add current request
        self.requests[client_id].append(now)
        return True


class AutoScaler:
    """Automatic scaling based on load"""

    def __init__(self, bot_manager: BotManager):
        self.bot_manager = bot_manager
        self.scaling_thresholds = {
            'scale_up_queue_size': 20,
            'scale_down_queue_size': 5,
            'scale_up_cpu': 80,
            'scale_down_cpu': 20,
            'min_bots': 1,
            'max_bots': 100
        }
        self.last_scale_time = time.time()
        self.scale_cooldown = 60  # seconds

    async def monitor_and_scale(self):
        """Monitor system and scale as needed"""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if time.time() - self.last_scale_time < self.scale_cooldown:
                    continue

                queue_size = self.bot_manager.search_queue.qsize()
                active_searches = len(self.bot_manager.active_searches)
                total_bots = len(self.bot_manager.bots)
                cpu_usage = psutil.cpu_percent()

                # Scale up conditions
                if (queue_size > self.scaling_thresholds['scale_up_queue_size'] or
                    cpu_usage > self.scaling_thresholds['scale_up_cpu']):

                    if total_bots < self.scaling_thresholds['max_bots']:
                        scale_amount = min(5, self.scaling_thresholds['max_bots'] - total_bots)
                        await self.bot_manager.scale_up(scale_amount)
                        self.last_scale_time = time.time()
                        logger.info(f"Scaled up to {total_bots + scale_amount} bots")

                # Scale down conditions
                elif (queue_size < self.scaling_thresholds['scale_down_queue_size'] and
                      cpu_usage < self.scaling_thresholds['scale_down_cpu'] and
                      active_searches < total_bots * 2):

                    if total_bots > self.scaling_thresholds['min_bots']:
                        scale_amount = min(2, total_bots - self.scaling_thresholds['min_bots'])
                        await self.bot_manager.scale_down(scale_amount)
                        self.last_scale_time = time.time()
                        logger.info(f"Scaled down to {total_bots - scale_amount} bots")

            except Exception as e:
                logger.error(f"AutoScaler error: {e}")


# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global bot_manager, auto_scaler

    logger.info("Starting YouTube Search Bot API Server")

    # Initialize bot manager
    bot_manager = BotManager()
    initial_bots = int(os.getenv('INITIAL_BOTS', '5'))
    await bot_manager.initialize(initial_bots)

    # Initialize auto-scaler
    auto_scaler = AutoScaler(bot_manager)
    asyncio.create_task(auto_scaler.monitor_and_scale())

    logger.info("API Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down API Server")
    if bot_manager:
        await bot_manager.cleanup()


# Create FastAPI app
app = FastAPI(
    title="YouTube Search Bot API",
    description="Scalable YouTube search automation with advanced anti-detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_status():
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return bot_manager.get_system_status()

# Submit search endpoint
@app.post("/search", response_model=SearchResponse)
async def submit_search(request: SearchRequest):
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        search_id = await bot_manager.submit_search(request)

        return SearchResponse(
            search_id=search_id,
            status="queued",
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get search results endpoint
@app.get("/search/{search_id}", response_model=SearchResponse)
async def get_search_results(search_id: str):
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    result = await bot_manager.get_search_status(search_id)

    if not result:
        raise HTTPException(status_code=404, detail="Search not found")

    return SearchResponse(**result)

# Batch search endpoint
@app.post("/search/batch")
async def batch_search(requests: List[SearchRequest]):
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    search_ids = []

    for request in requests:
        try:
            search_id = await bot_manager.submit_search(request)
            search_ids.append(search_id)
        except Exception as e:
            logger.error(f"Batch search error: {e}")
            search_ids.append(None)

    return {"search_ids": search_ids}

# Stream search results
@app.get("/search/{search_id}/stream")
async def stream_search_results(search_id: str):
    """Stream search results as they become available"""

    async def generate():
        max_wait = 300  # 5 minutes max wait
        start_time = time.time()

        while time.time() - start_time < max_wait:
            result = await bot_manager.get_search_status(search_id)

            if result:
                yield f"data: {json.dumps(result)}\n\n"

                if result['status'] in ['completed', 'failed']:
                    break

            await asyncio.sleep(1)

        yield "data: {\"status\": \"timeout\"}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# Admin endpoints
@app.post("/admin/scale")
async def manual_scale(action: str, count: int = 1):
    """Manually scale bots up or down"""
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if action == "up":
        await bot_manager.scale_up(count)
        return {"message": f"Scaled up by {count} bots"}
    elif action == "down":
        await bot_manager.scale_down(count)
        return {"message": f"Scaled down by {count} bots"}
    else:
        raise HTTPException(status_code=400, detail="Invalid action")

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get detailed system metrics"""
    if not bot_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return {
        "metrics": bot_manager.metrics,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": dict(psutil.virtual_memory()._asdict()),
            "disk": dict(psutil.disk_usage('/')._asdict()),
            "network": dict(psutil.net_io_counters()._asdict())
        },
        "queue_size": bot_manager.search_queue.qsize(),
        "active_searches": len(bot_manager.active_searches),
        "total_bots": len(bot_manager.bots)
    }


if __name__ == "__main__":
    # Run with uvicorn for production
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker since we handle concurrency internally
        log_level="info",
        access_log=True
    )