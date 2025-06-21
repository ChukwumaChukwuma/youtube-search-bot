import asyncio
import aiohttp
import random
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import json
import base64
from collections import defaultdict
import socket
import struct

logger = logging.getLogger(__name__)

class ProxyType(Enum):
    """Types of proxies"""
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"
    MOBILE = "mobile"
    STATIC_RESIDENTIAL = "static_residential"

class ProxyProtocol(Enum):
    """Proxy protocols"""
    HTTP = "http"
    HTTPS = "https"
    SOCKS5 = "socks5"

@dataclass
class ProxyServer:
    """Represents a proxy server"""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: ProxyType = ProxyType.RESIDENTIAL
    protocol: ProxyProtocol = ProxyProtocol.HTTP
    country: Optional[str] = None
    city: Optional[str] = None
    isp: Optional[str] = None
    last_used: float = 0
    success_count: int = 0
    failure_count: int = 0
    response_times: List[float] = None
    is_active: bool = True

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    def to_playwright_proxy(self) -> Dict[str, Any]:
        """Convert to Playwright proxy format"""
        proxy_url = f"{self.protocol.value}://"

        if self.username and self.password:
            proxy_url += f"{self.username}:{self.password}@"

        proxy_url += f"{self.host}:{self.port}"

        return {
            "server": proxy_url,
            "username": self.username,
            "password": self.password
        }

class ProxyRotationManager:
    """Manages residential proxy rotation with advanced features"""

    def __init__(self):
        self.proxy_pools: Dict[ProxyType, List[ProxyServer]] = {
            proxy_type: [] for proxy_type in ProxyType
        }
        self.active_proxies: Dict[str, ProxyServer] = {}
        self.blacklisted_proxies: List[ProxyServer] = []
        self.rotation_strategies = {
            'round_robin': self._round_robin_strategy,
            'least_used': self._least_used_strategy,
            'best_performance': self._best_performance_strategy,
            'random': self._random_strategy,
            'sticky_session': self._sticky_session_strategy,
        }
        self.current_strategy = 'best_performance'
        self.rotation_index = defaultdict(int)
        self.session_proxies: Dict[str, ProxyServer] = {}
        self.proxy_providers = []
        self.health_check_interval = 300  # 5 minutes
        self.max_failures = 3
        self.min_success_rate = 0.7
        self.max_response_time = 5.0

    async def initialize(self):
        """Initialize proxy rotation manager"""
        logger.info("Initializing Proxy Rotation Manager")

        # Load proxy providers
        await self._load_proxy_providers()

        # Fetch initial proxy list
        await self._fetch_proxies()

        # Start health check task
        asyncio.create_task(self._health_check_loop())

        logger.info(f"Initialized with {sum(len(pool) for pool in self.proxy_pools.values())} proxies")

    async def _load_proxy_providers(self):
        """Load proxy provider configurations"""
        # In production, this would load from config file or environment
        self.proxy_providers = [
            {
                'name': 'residential_provider_1',
                'api_endpoint': 'https://api.residentialproxies.com/v1/proxies',
                'api_key': 'YOUR_API_KEY',
                'proxy_type': ProxyType.RESIDENTIAL,
                'countries': ['US', 'UK', 'CA', 'AU', 'DE', 'FR'],
                'pool_size': 1000,
            },
            {
                'name': 'mobile_provider_1',
                'api_endpoint': 'https://api.mobileproxies.com/v1/proxies',
                'api_key': 'YOUR_API_KEY',
                'proxy_type': ProxyType.MOBILE,
                'countries': ['US', 'UK'],
                'pool_size': 500,
            }
        ]

    async def _fetch_proxies(self):
        """Fetch proxies from providers"""
        # In production, this would make actual API calls to proxy providers
        # For now, we'll simulate with realistic proxy data

        # Simulate residential proxies
        for i in range(100):
            proxy = ProxyServer(
                host=f"res-proxy-{i}.residential.com",
                port=random.choice([8080, 8888, 3128, 9999]),
                username=f"user_{i}",
                password=f"pass_{i}",
                proxy_type=ProxyType.RESIDENTIAL,
                protocol=ProxyProtocol.HTTP,
                country=random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR']),
                city=random.choice(['New York', 'London', 'Toronto', 'Sydney', 'Berlin', 'Paris']),
                isp=random.choice(['Comcast', 'AT&T', 'Verizon', 'BT', 'Rogers', 'Telstra'])
            )
            self.proxy_pools[ProxyType.RESIDENTIAL].append(proxy)

        # Simulate mobile proxies
        for i in range(50):
            proxy = ProxyServer(
                host=f"mobile-proxy-{i}.4g.com",
                port=random.choice([8080, 8888]),
                username=f"mobile_user_{i}",
                password=f"mobile_pass_{i}",
                proxy_type=ProxyType.MOBILE,
                protocol=ProxyProtocol.SOCKS5,
                country=random.choice(['US', 'UK']),
                city=random.choice(['Los Angeles', 'Manchester']),
                isp=random.choice(['T-Mobile', 'Verizon', 'EE', 'O2'])
            )
            self.proxy_pools[ProxyType.MOBILE].append(proxy)

    async def get_proxy(self, session_id: Optional[str] = None, 
                       proxy_type: Optional[ProxyType] = None,
                       country: Optional[str] = None) -> Optional[ProxyServer]:
        """Get a proxy based on rotation strategy"""
        # Filter proxies based on requirements
        available_proxies = self._filter_proxies(proxy_type, country)

        if not available_proxies:
            logger.warning("No available proxies matching criteria")
            return None

        # Apply rotation strategy
        strategy_func = self.rotation_strategies.get(self.current_strategy, self._random_strategy)
        proxy = await strategy_func(available_proxies, session_id)

        if proxy:
            # Mark proxy as active
            proxy.last_used = time.time()
            self.active_proxies[f"{proxy.host}:{proxy.port}"] = proxy

            # Track session if provided
            if session_id and self.current_strategy == 'sticky_session':
                self.session_proxies[session_id] = proxy

        return proxy

    def _filter_proxies(self, proxy_type: Optional[ProxyType] = None,
                       country: Optional[str] = None) -> List[ProxyServer]:
        """Filter proxies based on criteria"""
        proxies = []

        # Get proxies of specified type or all types
        if proxy_type:
            proxies = self.proxy_pools.get(proxy_type, [])
        else:
            for pool in self.proxy_pools.values():
                proxies.extend(pool)

        # Filter by country if specified
        if country:
            proxies = [p for p in proxies if p.country == country]

        # Filter out inactive and blacklisted proxies
        proxies = [
            p for p in proxies 
            if p.is_active and p not in self.blacklisted_proxies
        ]

        # Filter by performance metrics
        proxies = [
            p for p in proxies
            if p.success_rate >= self.min_success_rate or p.success_count + p.failure_count < 10
        ]

        return proxies

    async def _round_robin_strategy(self, proxies: List[ProxyServer], 
                                   session_id: Optional[str] = None) -> Optional[ProxyServer]:
        """Round-robin rotation strategy"""
        if not proxies:
            return None

        key = 'global' if not session_id else session_id
        index = self.rotation_index[key] % len(proxies)
        self.rotation_index[key] += 1

        return proxies[index]

    async def _least_used_strategy(self, proxies: List[ProxyServer],
                                  session_id: Optional[str] = None) -> Optional[ProxyServer]:
        """Select least recently used proxy"""
        return min(proxies, key=lambda p: p.last_used) if proxies else None

    async def _best_performance_strategy(self, proxies: List[ProxyServer],
                                       session_id: Optional[str] = None) -> Optional[ProxyServer]:
        """Select proxy with best performance metrics"""
        if not proxies:
            return None

        # Score proxies based on success rate and response time
        def score_proxy(proxy: ProxyServer) -> float:
            success_weight = 0.7
            speed_weight = 0.3

            # Calculate success score (0-1)
            success_score = proxy.success_rate if proxy.success_count + proxy.failure_count >= 5 else 0.8

            # Calculate speed score (0-1)
            if proxy.response_times:
                avg_time = proxy.average_response_time
                speed_score = max(0, 1 - (avg_time / self.max_response_time))
            else:
                speed_score = 0.5

            return success_weight * success_score + speed_weight * speed_score

        # Sort by score and add some randomness to top performers
        sorted_proxies = sorted(proxies, key=score_proxy, reverse=True)
        top_count = min(5, len(sorted_proxies))

        return random.choice(sorted_proxies[:top_count])

    async def _random_strategy(self, proxies: List[ProxyServer],
                             session_id: Optional[str] = None) -> Optional[ProxyServer]:
        """Random proxy selection"""
        return random.choice(proxies) if proxies else None

    async def _sticky_session_strategy(self, proxies: List[ProxyServer],
                                     session_id: Optional[str] = None) -> Optional[ProxyServer]:
        """Sticky session - same proxy for same session"""
        if session_id and session_id in self.session_proxies:
            proxy = self.session_proxies[session_id]
            if proxy.is_active and proxy not in self.blacklisted_proxies:
                return proxy

        # Fall back to best performance for new session
        return await self._best_performance_strategy(proxies, session_id)

    async def report_proxy_result(self, proxy: ProxyServer, success: bool,
                                response_time: Optional[float] = None):
        """Report proxy usage result"""
        if success:
            proxy.success_count += 1
            if response_time:
                proxy.response_times.append(response_time)
                # Keep only last 100 response times
                if len(proxy.response_times) > 100:
                    proxy.response_times = proxy.response_times[-100:]
        else:
            proxy.failure_count += 1

        # Check if proxy should be blacklisted
        if proxy.failure_count >= self.max_failures:
            failure_rate = proxy.failure_count / (proxy.success_count + proxy.failure_count)
            if failure_rate > 0.5:
                await self._blacklist_proxy(proxy)

    async def _blacklist_proxy(self, proxy: ProxyServer):
        """Blacklist a poorly performing proxy"""
        logger.warning(f"Blacklisting proxy {proxy.host}:{proxy.port} due to poor performance")
        proxy.is_active = False
        self.blacklisted_proxies.append(proxy)

        # Remove from active proxies
        proxy_key = f"{proxy.host}:{proxy.port}"
        if proxy_key in self.active_proxies:
            del self.active_proxies[proxy_key]

        # Remove from session proxies
        for session_id, session_proxy in list(self.session_proxies.items()):
            if session_proxy == proxy:
                del self.session_proxies[session_id]

    async def _health_check_loop(self):
        """Periodically check proxy health"""
        while True:
            await asyncio.sleep(self.health_check_interval)
            await self._check_proxy_health()

    async def _check_proxy_health(self):
        """Check health of all proxies"""
        logger.info("Starting proxy health check")

        tasks = []
        for proxy_type, proxies in self.proxy_pools.items():
            for proxy in proxies:
                if proxy.is_active:
                    task = self._test_proxy(proxy)
                    tasks.append(task)

        # Run health checks with concurrency limit
        semaphore = asyncio.Semaphore(50)
        async def limited_test(proxy):
            async with semaphore:
                return await self._test_proxy(proxy)

        results = await asyncio.gather(*[limited_test(p) for p in proxies], return_exceptions=True)

        # Process results
        healthy_count = sum(1 for r in results if r is True)
        logger.info(f"Health check complete: {healthy_count}/{len(results)} proxies healthy")

    async def _test_proxy(self, proxy: ProxyServer) -> bool:
        """Test if a proxy is working"""
        test_url = "http://httpbin.org/ip"
        timeout = aiohttp.ClientTimeout(total=10)

        proxy_url = f"{proxy.protocol.value}://"
        if proxy.username and proxy.password:
            proxy_url += f"{proxy.username}:{proxy.password}@"
        proxy_url += f"{proxy.host}:{proxy.port}"

        try:
            start_time = time.time()

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(test_url, proxy=proxy_url) as response:
                    if response.status == 200:
                        response_time = time.time() - start_time
                        await self.report_proxy_result(proxy, True, response_time)
                        return True
                    else:
                        await self.report_proxy_result(proxy, False)
                        return False

        except Exception as e:
            logger.debug(f"Proxy test failed for {proxy.host}:{proxy.port}: {e}")
            await self.report_proxy_result(proxy, False)
            return False

    def set_rotation_strategy(self, strategy: str):
        """Set proxy rotation strategy"""
        if strategy in self.rotation_strategies:
            self.current_strategy = strategy
            logger.info(f"Set rotation strategy to: {strategy}")
        else:
            logger.warning(f"Unknown rotation strategy: {strategy}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get proxy pool statistics"""
        stats = {
            'total_proxies': sum(len(pool) for pool in self.proxy_pools.values()),
            'active_proxies': len([p for pool in self.proxy_pools.values() for p in pool if p.is_active]),
            'blacklisted_proxies': len(self.blacklisted_proxies),
            'current_strategy': self.current_strategy,
            'pools': {}
        }

        for proxy_type, pool in self.proxy_pools.items():
            active_pool = [p for p in pool if p.is_active]
            stats['pools'][proxy_type.value] = {
                'total': len(pool),
                'active': len(active_pool),
                'average_success_rate': sum(p.success_rate for p in active_pool) / len(active_pool) if active_pool else 0,
                'countries': list(set(p.country for p in active_pool if p.country)),
            }

        return stats

    async def refresh_proxy_pool(self):
        """Refresh the proxy pool with new proxies"""
        logger.info("Refreshing proxy pool")

        # Clear blacklist if it's getting too large
        if len(self.blacklisted_proxies) > 100:
            self.blacklisted_proxies = self.blacklisted_proxies[-50:]

        # Fetch new proxies
        await self._fetch_proxies()

        # Remove old inactive proxies
        for proxy_type, pool in self.proxy_pools.items():
            self.proxy_pools[proxy_type] = [
                p for p in pool 
                if p.is_active or time.time() - p.last_used < 3600  # Keep proxies used within last hour
            ]

    def export_proxy_stats(self, filepath: str):
        """Export proxy statistics to file"""
        stats = {
            'timestamp': time.time(),
            'statistics': self.get_statistics(),
            'proxy_performance': []
        }

        for pool in self.proxy_pools.values():
            for proxy in pool:
                stats['proxy_performance'].append({
                    'host': proxy.host,
                    'port': proxy.port,
                    'type': proxy.proxy_type.value,
                    'country': proxy.country,
                    'success_rate': proxy.success_rate,
                    'avg_response_time': proxy.average_response_time,
                    'total_uses': proxy.success_count + proxy.failure_count,
                    'is_active': proxy.is_active
                })

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)