"""
YouTube Search Bot Python SDK
A powerful Python client for the YouTube Search Bot API with full feature support
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from urllib.parse import urljoin
import backoff
from datetime import datetime

logger = logging.getLogger(__name__)

class SearchStatus(Enum):
    """Search status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class SearchRequest:
    """Search request parameters"""
    keyword: str
    max_results: int = 50
    session_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for API request"""
        data = {
            "keyword": self.keyword,
            "max_results": self.max_results
        }
        if self.session_id:
            data["session_id"] = self.session_id
        if self.options:
            data["options"] = self.options
        return data

@dataclass
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    channel: str
    views: str
    duration: str
    search_keyword: str
    timestamp: str

    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchResult':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class SearchResponse:
    """Search response from API"""
    search_id: str
    status: SearchStatus
    results: Optional[List[SearchResult]] = None
    error: Optional[str] = None
    timestamp: str = ""
    duration_ms: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchResponse':
        """Create from dictionary"""
        response = cls(
            search_id=data["search_id"],
            status=SearchStatus(data["status"]),
            error=data.get("error"),
            timestamp=data.get("timestamp", ""),
            duration_ms=data.get("duration_ms")
        )

        if data.get("results"):
            response.results = [SearchResult.from_dict(r) for r in data["results"]]

        return response

@dataclass
class SystemStatus:
    """System status information"""
    status: str
    active_searches: int
    total_browsers: int
    cpu_usage: float
    memory_usage: float
    uptime_seconds: float
    total_searches_completed: int
    average_search_time_ms: float
    success_rate: float

    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemStatus':
        """Create from dictionary"""
        return cls(**data)

class YouTubeSearchBotError(Exception):
    """Base exception for YouTube Search Bot SDK"""
    pass

class RateLimitError(YouTubeSearchBotError):
    """Rate limit exceeded error"""
    pass

class SearchTimeoutError(YouTubeSearchBotError):
    """Search timeout error"""
    pass

class YouTubeSearchBotClient:
    """
    YouTube Search Bot API Client

    Example usage:
        async with YouTubeSearchBotClient("http://localhost:8000") as client:
            results = await client.search("python programming", max_results=10)
            for result in results:
                print(f"{result.title} - {result.url}")
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: int = 300, max_retries: int = 3):
        """
        Initialize the client

        Args:
            base_url: Base URL of the API server
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_session(self):
        """Ensure aiohttp session exists"""
        if self.session is None or self.session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )

    async def close(self):
        """Close the client session"""
        if self.session and not self.session.closed:
            await self.session.close()
        self._closed = True

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3,
        max_time=60
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with retry logic"""
        await self._ensure_session()

        url = urljoin(self.base_url, endpoint)

        async with self.session.request(method, url, **kwargs) as response:
            data = await response.json()

            if response.status == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status >= 400:
                error_msg = data.get("detail", f"HTTP {response.status}")
                raise YouTubeSearchBotError(f"API error: {error_msg}")

            return data

    async def search(self, keyword: str, max_results: int = 50,
                    session_id: Optional[str] = None,
                    options: Optional[Dict[str, Any]] = None,
                    wait_for_results: bool = True,
                    polling_interval: float = 1.0) -> List[SearchResult]:
        """
        Perform a YouTube search

        Args:
            keyword: Search keyword
            max_results: Maximum number of results to return
            session_id: Optional session ID for tracking
            options: Additional search options
            wait_for_results: Whether to wait for results or return immediately
            polling_interval: Interval for polling results if waiting

        Returns:
            List of search results

        Raises:
            YouTubeSearchBotError: If search fails
            SearchTimeoutError: If search times out
        """
        request = SearchRequest(
            keyword=keyword,
            max_results=max_results,
            session_id=session_id,
            options=options
        )

        # Submit search
        response_data = await self._request(
            "POST",
            "/search",
            json=request.to_dict()
        )

        search_response = SearchResponse.from_dict(response_data)

        if not wait_for_results:
            return []

        # Poll for results
        start_time = time.time()
        max_wait = self.timeout.total or 300

        while time.time() - start_time < max_wait:
            # Get search status
            status_data = await self._request(
                "GET",
                f"/search/{search_response.search_id}"
            )

            search_response = SearchResponse.from_dict(status_data)

            if search_response.status == SearchStatus.COMPLETED:
                return search_response.results or []
            elif search_response.status == SearchStatus.FAILED:
                raise YouTubeSearchBotError(f"Search failed: {search_response.error}")
            elif search_response.status == SearchStatus.TIMEOUT:
                raise SearchTimeoutError("Search timed out")

            await asyncio.sleep(polling_interval)

        raise SearchTimeoutError(f"Search timed out after {max_wait} seconds")

    async def search_async(self, keyword: str, max_results: int = 50,
                          session_id: Optional[str] = None,
                          options: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a search without waiting for results

        Returns:
            Search ID for checking status later
        """
        request = SearchRequest(
            keyword=keyword,
            max_results=max_results,
            session_id=session_id,
            options=options
        )

        response_data = await self._request(
            "POST",
            "/search",
            json=request.to_dict()
        )

        return response_data["search_id"]

    async def get_search_results(self, search_id: str) -> SearchResponse:
        """
        Get results for a previously submitted search

        Args:
            search_id: The search ID returned by search_async

        Returns:
            SearchResponse with current status and results if available
        """
        response_data = await self._request(
            "GET",
            f"/search/{search_id}"
        )

        return SearchResponse.from_dict(response_data)

    async def batch_search(self, keywords: List[str], max_results: int = 50,
                          session_id: Optional[str] = None) -> Dict[str, List[SearchResult]]:
        """
        Perform multiple searches in batch

        Args:
            keywords: List of keywords to search
            max_results: Maximum results per search
            session_id: Optional session ID

        Returns:
            Dictionary mapping keywords to their results
        """
        # Submit all searches
        requests = [
            SearchRequest(keyword=kw, max_results=max_results, session_id=session_id)
            for kw in keywords
        ]

        response_data = await self._request(
            "POST",
            "/search/batch",
            json=[r.to_dict() for r in requests]
        )

        search_ids = response_data["search_ids"]

        # Wait for all results
        results = {}
        for keyword, search_id in zip(keywords, search_ids):
            if search_id:
                try:
                    search_results = await self.search(
                        keyword=keyword,
                        max_results=max_results,
                        session_id=session_id,
                        wait_for_results=True
                    )
                    results[keyword] = search_results
                except Exception as e:
                    logger.error(f"Batch search failed for '{keyword}': {e}")
                    results[keyword] = []
            else:
                results[keyword] = []

        return results

    async def stream_search_results(self, search_id: str) -> AsyncIterator[SearchResponse]:
        """
        Stream search results as they become available

        Args:
            search_id: The search ID to stream

        Yields:
            SearchResponse objects with current status
        """
        await self._ensure_session()

        url = urljoin(self.base_url, f"/search/{search_id}/stream")

        async with self.session.get(url) as response:
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield SearchResponse.from_dict(data)

                    if data["status"] in ["completed", "failed", "timeout"]:
                        break

    async def get_status(self) -> SystemStatus:
        """
        Get system status

        Returns:
            SystemStatus object with current system information
        """
        response_data = await self._request("GET", "/status")
        return SystemStatus.from_dict(response_data)

    async def get_health(self) -> bool:
        """
        Check API health

        Returns:
            True if API is healthy
        """
        try:
            response_data = await self._request("GET", "/health")
            return response_data.get("status") == "healthy"
        except:
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed system metrics

        Returns:
            Dictionary with system metrics
        """
        return await self._request("GET", "/metrics")

    async def scale(self, action: str, count: int = 1) -> Dict[str, str]:
        """
        Manually scale the system up or down

        Args:
            action: "up" or "down"
            count: Number of bots to add/remove

        Returns:
            Response message
        """
        return await self._request(
            "POST",
            "/admin/scale",
            params={"action": action, "count": count}
        )


class YouTubeSearchBotAsync:
    """
    Simplified async interface for YouTube Search Bot

    Example:
        bot = YouTubeSearchBotAsync("http://localhost:8000")
        results = await bot.search("python tutorials")
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.client = YouTubeSearchBotClient(base_url, api_key)

    async def search(self, keyword: str, max_results: int = 50) -> List[SearchResult]:
        """Simple search interface"""
        async with self.client as client:
            return await client.search(keyword, max_results)

    async def batch_search(self, keywords: List[str], max_results: int = 50) -> Dict[str, List[SearchResult]]:
        """Batch search interface"""
        async with self.client as client:
            return await client.batch_search(keywords, max_results)

    async def get_status(self) -> SystemStatus:
        """Get system status"""
        async with self.client as client:
            return await client.get_status()


def create_client(base_url: str, api_key: Optional[str] = None) -> YouTubeSearchBotClient:
    """
    Factory function to create a client

    Args:
        base_url: API base URL
        api_key: Optional API key

    Returns:
        YouTubeSearchBotClient instance
    """
    return YouTubeSearchBotClient(base_url, api_key)


# Synchronous wrapper for those who prefer sync code
class YouTubeSearchBot:
    """
    Synchronous wrapper for the async client

    Example:
        bot = YouTubeSearchBot("http://localhost:8000")
        results = bot.search("python programming")
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.async_client = YouTubeSearchBotAsync(base_url, api_key)

    def search(self, keyword: str, max_results: int = 50) -> List[SearchResult]:
        """Synchronous search"""
        return asyncio.run(self.async_client.search(keyword, max_results))

    def batch_search(self, keywords: List[str], max_results: int = 50) -> Dict[str, List[SearchResult]]:
        """Synchronous batch search"""
        return asyncio.run(self.async_client.batch_search(keywords, max_results))

    def get_status(self) -> SystemStatus:
        """Get system status synchronously"""
        return asyncio.run(self.async_client.get_status())


# Example usage
if __name__ == "__main__":
    async def example():
        # Create client
        async with YouTubeSearchBotClient("http://localhost:8000") as client:
            # Simple search
            print("Searching for 'python tutorials'...")
            results = await client.search("python tutorials", max_results=5)

            for i, result in enumerate(results, 1):
                print(f"{i}. {result.title}")
                print(f"   URL: {result.url}")
                print(f"   Channel: {result.channel}")
                print()

            # Check system status
            status = await client.get_status()
            print(f"System Status: {status.status}")
            print(f"Active Searches: {status.active_searches}")
            print(f"Success Rate: {status.success_rate:.2%}")

    # Run example
    asyncio.run(example())