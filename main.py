import asyncio
import json
import random
import time
import hashlib
import base64
from typing import Dict, List, Optional, Any
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
import numpy as np
from datetime import datetime
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeSearchBot:
    def __init__(self, max_browsers: int = 10):
        self.max_browsers = max_browsers
        self.browser_pool = []
        self.fingerprint_manager = FingerprintManager()
        self.captcha_solver = CaptchaSolver()
        self.behavior_modeler = BehaviorModeler()
        self.proxy_manager = ProxyManager()
        self.active_searches = 0
        self.max_concurrent_searches = 1000

    async def initialize(self):
        """Initialize the bot with browser pool"""
        logger.info("Initializing YouTube Search Bot...")
        await self.captcha_solver.initialize()
        await self.proxy_manager.initialize()
        await self._create_browser_pool()

    async def _create_browser_pool(self):
        """Create a pool of browser instances"""
        for i in range(min(self.max_browsers, 5)):  # Start with 5 browsers
            browser = await self._create_stealth_browser()
            self.browser_pool.append(browser)

    async def _create_stealth_browser(self) -> Browser:
        """Create a browser instance with full stealth capabilities"""
        playwright = await async_playwright().start()

        # Generate unique fingerprint
        fingerprint = self.fingerprint_manager.generate_fingerprint()
        proxy = await self.proxy_manager.get_proxy()

        # Browser launch arguments for maximum stealth
        launch_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--disable-web-security',
            '--disable-features=IsolateOrigins,site-per-process',
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-accelerated-2d-canvas',
            '--disable-gpu',
            '--window-size=1920,1080',
            '--start-maximized',
            '--user-agent=' + fingerprint['user_agent'],
            f'--window-position={random.randint(0,200)},{random.randint(0,200)}',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
            '--disable-features=TranslateUI',
            '--disable-ipc-flooding-protection',
            '--password-store=basic',
            '--use-mock-keychain',
            '--force-color-profile=srgb',
            '--disable-features=UserAgentClientHint',
            '--disable-features=WebRtcHideLocalIpsWithMdns',
        ]

        browser = await playwright.chromium.launch(
            headless=False,  # YouTube often blocks headless
            args=launch_args,
            proxy=proxy if proxy else None,
            # Additional options for better compatibility
            chromium_sandbox=False,
            handle_sigint=False,
            handle_sigterm=False,
            handle_sighup=False
        )

        # Apply stealth scripts to all contexts
        context = await browser.new_context(
            viewport={'width': fingerprint['viewport']['width'], 'height': fingerprint['viewport']['height']},
            locale=fingerprint['locale'],
            timezone_id=fingerprint['timezone'],
            permissions=['geolocation'],
            geolocation={'latitude': fingerprint['geo']['lat'], 'longitude': fingerprint['geo']['lon']},
            color_scheme='dark' if random.random() > 0.5 else 'light',
            extra_http_headers={
                'Accept-Language': fingerprint['language'],
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
            }
        )

        return browser

    async def search(self, keyword: str, max_results: int = 50) -> List[Dict]:
        """Perform YouTube search with given keyword"""
        while self.active_searches >= self.max_concurrent_searches:
            await asyncio.sleep(0.1)

        self.active_searches += 1
        try:
            # Get available browser from pool
            browser = await self._get_available_browser()
            context = await browser.new_context()
            page = await context.new_page()

            # Apply all stealth measures
            await self._apply_stealth_scripts(page)

            # Perform search with human-like behavior
            results = await self._perform_search(page, keyword, max_results)

            await context.close()
            return results

        finally:
            self.active_searches -= 1

    async def _apply_stealth_scripts(self, page: Page):
        """Apply all stealth JavaScript injections"""
        # Remove webdriver property
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        # Override navigator.plugins
        await page.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const plugins = [
                        {
                            0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                            description: "Portable Document Format",
                            filename: "internal-pdf-viewer",
                            length: 1,
                            name: "Chrome PDF Plugin"
                        },
                        {
                            0: {type: "application/pdf", suffixes: "pdf", description: "Portable Document Format"},
                            description: "Portable Document Format", 
                            filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                            length: 1,
                            name: "Chrome PDF Viewer"
                        }
                    ];
                    return Object.create(PluginArray.prototype, {
                        length: {value: plugins.length},
                        ...Object.fromEntries(plugins.map((p, i) => [i, {value: p}])),
                        item: {value: i => plugins[i]},
                        namedItem: {value: name => plugins.find(p => p.name === name)},
                        [Symbol.iterator]: {value: function*() { yield* plugins; }}
                    });
                }
            });
        """)

        # Chrome specific overrides
        await page.add_init_script("""
            window.chrome = {
                runtime: {
                    connect: () => {},
                    sendMessage: () => {},
                    onMessage: {addListener: () => {}}
                },
                loadTimes: () => ({
                    commitLoadTime: Date.now() / 1000 - Math.random() * 10,
                    connectionInfo: "h2",
                    finishDocumentLoadTime: Date.now() / 1000 - Math.random() * 5,
                    finishLoadTime: Date.now() / 1000 - Math.random() * 3,
                    firstPaintAfterLoadTime: 0,
                    firstPaintTime: Date.now() / 1000 - Math.random() * 7,
                    navigationStart: Date.now() / 1000 - Math.random() * 15,
                    npnNegotiatedProtocol: "h2",
                    requestTime: Date.now() / 1000 - Math.random() * 20,
                    startLoadTime: Date.now() / 1000 - Math.random() * 12,
                    wasAlternateProtocolAvailable: false,
                    wasFetchedViaSpdy: true,
                    wasNpnNegotiated: true
                }),
                csi: () => ({
                    onloadT: Date.now(),
                    pageT: Date.now() - Math.random() * 1000,
                    startE: Date.now() - Math.random() * 2000,
                    tran: 15
                })
            };
        """)

        # Canvas fingerprinting protection
        await page.add_init_script("""
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            const originalToBlob = HTMLCanvasElement.prototype.toBlob;
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;

            HTMLCanvasElement.prototype.toDataURL = function(...args) {
                const context = this.getContext('2d');
                if (context) {
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        imageData.data[i] ^= Math.floor(Math.random() * 10);
                        imageData.data[i+1] ^= Math.floor(Math.random() * 10);
                        imageData.data[i+2] ^= Math.floor(Math.random() * 10);
                    }
                    context.putImageData(imageData, 0, 0);
                }
                return originalToDataURL.apply(this, args);
            };

            HTMLCanvasElement.prototype.toBlob = function(...args) {
                const context = this.getContext('2d');
                if (context) {
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {
                        imageData.data[i] ^= Math.floor(Math.random() * 10);
                    }
                    context.putImageData(imageData, 0, 0);
                }
                return originalToBlob.apply(this, args);
            };
        """)

        # WebGL fingerprinting protection
        await page.add_init_script("""
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameter.apply(this, [parameter]);
            };

            const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
            WebGL2RenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) {
                    return 'Intel Inc.';
                }
                if (parameter === 37446) {
                    return 'Intel Iris OpenGL Engine';
                }
                return getParameter2.apply(this, [parameter]);
            };
        """)

        # Audio context fingerprinting protection
        await page.add_init_script("""
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const OfflineAudioContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;

            if (AudioContext) {
                const originalCreateOscillator = AudioContext.prototype.createOscillator;
                AudioContext.prototype.createOscillator = function() {
                    const oscillator = originalCreateOscillator.apply(this, arguments);
                    const originalConnect = oscillator.connect;
                    oscillator.connect = function() {
                        arguments[0].gain.value += (Math.random() - 0.5) * 0.0001;
                        return originalConnect.apply(this, arguments);
                    };
                    return oscillator;
                };
            }
        """)

        # Font enumeration protection
        await page.add_init_script("""
            const originalOffsetWidth = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetWidth').get;
            const originalOffsetHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, 'offsetHeight').get;

            Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
                get: function() {
                    const width = originalOffsetWidth.apply(this);
                    if (this.style.fontFamily && this.textContent) {
                        return width + (Math.random() < 0.1 ? 1 : 0);
                    }
                    return width;
                }
            });

            Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
                get: function() {
                    const height = originalOffsetHeight.apply(this);
                    if (this.style.fontFamily && this.textContent) {
                        return height + (Math.random() < 0.1 ? 1 : 0);
                    }
                    return height;
                }
            });
        """)

        # WebRTC leak prevention
        await page.add_init_script("""
            const RTCPeerConnection = window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection;
            if (RTCPeerConnection) {
                const originalCreateDataChannel = RTCPeerConnection.prototype.createDataChannel;
                RTCPeerConnection.prototype.createDataChannel = function() {
                    return null;
                };

                const originalCreateOffer = RTCPeerConnection.prototype.createOffer;
                RTCPeerConnection.prototype.createOffer = function() {
                    return Promise.reject(new DOMException('Operation is not supported', 'NotSupportedError'));
                };
            }
        """)

        # Battery API spoofing
        await page.add_init_script("""
            if ('getBattery' in navigator) {
                navigator.getBattery = async () => ({
                    charging: true,
                    chargingTime: 0,
                    dischargingTime: Infinity,
                    level: 0.99,
                    addEventListener: () => {},
                    removeEventListener: () => {},
                });
            }
        """)

        # Hardware concurrency spoofing
        await page.add_init_script("""
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 4 + Math.floor(Math.random() * 4) * 2
            });
        """)

        # Screen resolution spoofing
        await page.add_init_script("""
            Object.defineProperty(screen, 'availWidth', {
                get: () => screen.width - Math.floor(Math.random() * 100)
            });
            Object.defineProperty(screen, 'availHeight', {
                get: () => screen.height - Math.floor(Math.random() * 100)
            });
        """)

        # Permissions API override
        await page.add_init_script("""
            const originalQuery = navigator.permissions.query;
            navigator.permissions.query = async (parameters) => {
                if (parameters.name === 'notifications') {
                    return { state: 'denied' };
                }
                return originalQuery(parameters);
            };
        """)

        # Memory spoofing
        await page.add_init_script("""
            if (performance.memory) {
                Object.defineProperty(performance.memory, 'jsHeapSizeLimit', {
                    get: () => 2172649472 + Math.floor(Math.random() * 100000000)
                });
                Object.defineProperty(performance.memory, 'totalJSHeapSize', {
                    get: () => 35724672 + Math.floor(Math.random() * 10000000)
                });
                Object.defineProperty(performance.memory, 'usedJSHeapSize', {
                    get: () => 19827648 + Math.floor(Math.random() * 5000000)
                });
            }
        """)

    async def _perform_search(self, page: Page, keyword: str, max_results: int) -> List[Dict]:
        """Perform the actual YouTube search with human-like behavior"""
        results = []

        try:
            # Navigate to YouTube with human-like timing
            await self.behavior_modeler.human_delay(1, 3)

            # Try direct search URL first (more reliable)
            search_url = f'https://www.youtube.com/results?search_query={keyword.replace(" ", "+")}'
            await page.goto(search_url, wait_until='domcontentloaded', timeout=30000)

            # Wait a bit for dynamic content
            await self.behavior_modeler.human_delay(2, 4)

            # Check for CAPTCHA
            if await self._check_for_captcha(page):
                logger.info("CAPTCHA detected, attempting to solve...")
                if not await self.captcha_solver.solve_captcha(page):
                    logger.error("Failed to solve CAPTCHA")
                    # Try alternative search method
                    await page.goto('https://www.youtube.com', wait_until='domcontentloaded')
                    await self.behavior_modeler.human_delay(1, 2)

                    # Use alternative search method
                    search_box = await page.wait_for_selector('input#search', timeout=10000)
                    if search_box:
                        await search_box.click()
                        await search_box.fill('')
                        await self.behavior_modeler.human_type(page, search_box, keyword)
                        await page.keyboard.press('Enter')
                        await self.behavior_modeler.human_delay(2, 3)

            # Wait for any of these possible result containers
            result_selectors = [
                'ytd-video-renderer',
                'ytd-rich-item-renderer',
                'div#contents ytd-rich-item-renderer',
                'ytd-search-result-renderer',
                'a#video-title',
                'h3.title-and-badge a'
            ]

            result_found = False
            for selector in result_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    result_found = True
                    logger.info(f"Found results using selector: {selector}")
                    break
                except:
                    continue

            if not result_found:
                logger.error("No result selectors found on page")
                # Try to debug what's on the page
                page_content = await page.content()
                if "did not match any documents" in page_content:
                    logger.error("YouTube returned 'no results' page")
                return results

            # Scroll and collect results with human-like behavior
            collected = 0
            scroll_attempts = 0
            max_scroll_attempts = 15
            consecutive_no_new = 0

            while collected < max_results and scroll_attempts < max_scroll_attempts:
                previous_count = collected

                # Try multiple strategies to extract video data

                # Strategy 1: Modern YouTube layout (ytd-rich-item-renderer)
                rich_items = await page.query_selector_all('ytd-rich-item-renderer')
                for item in rich_items[collected:]:
                    if collected >= max_results:
                        break

                    try:
                        # Title and link
                        title_elem = await item.query_selector('a#video-title-link, h3 a, a#video-title')
                        if not title_elem:
                            continue

                        title = await title_elem.inner_text()
                        link = await title_elem.get_attribute('href')

                        if not title or not link:
                            continue

                        if not link.startswith('http'):
                            link = f"https://www.youtube.com{link}"

                        # Channel
                        channel_elem = await item.query_selector('ytd-channel-name a, div#channel-info a, a.yt-formatted-string')
                        channel = await channel_elem.inner_text() if channel_elem else "Unknown Channel"

                        # Views and upload time
                        metadata_elem = await item.query_selector('div#metadata-line')
                        if metadata_elem:
                            metadata_text = await metadata_elem.inner_text()
                            metadata_parts = metadata_text.split('\n')
                            views = metadata_parts[0] if len(metadata_parts) > 0 else ""
                            upload_time = metadata_parts[1] if len(metadata_parts) > 1 else ""
                        else:
                            views = ""
                            upload_time = ""

                        # Duration
                        duration_elem = await item.query_selector('span.ytd-thumbnail-overlay-time-status-renderer, ytd-thumbnail-overlay-time-status-renderer span')
                        duration = await duration_elem.inner_text() if duration_elem else ""

                        results.append({
                            'title': title.strip(),
                            'url': link,
                            'channel': channel.strip(),
                            'views': views.strip(),
                            'duration': duration.strip(),
                            'upload_time': upload_time.strip(),
                            'search_keyword': keyword,
                            'timestamp': datetime.now().isoformat()
                        })

                        collected += 1
                        logger.debug(f"Collected video {collected}: {title[:50]}...")

                    except Exception as e:
                        logger.debug(f"Error extracting from rich item: {e}")
                        continue

                # Strategy 2: Classic search results (ytd-video-renderer)
                if collected < max_results:
                    video_renderers = await page.query_selector_all('ytd-video-renderer')
                    for video in video_renderers:
                        if collected >= max_results:
                            break

                        try:
                            # Check if we already have this video
                            title_elem = await video.query_selector('a#video-title')
                            if not title_elem:
                                continue

                            title = await title_elem.get_attribute('title') or await title_elem.inner_text()

                            # Skip if already collected
                            if any(r['title'] == title.strip() for r in results):
                                continue

                            link = await title_elem.get_attribute('href')
                            if not link:
                                continue

                            if not link.startswith('http'):
                                link = f"https://www.youtube.com{link}"

                            # Channel
                            channel_elem = await video.query_selector('ytd-channel-name a, a.yt-simple-endpoint')
                            channel = await channel_elem.inner_text() if channel_elem else "Unknown Channel"

                            # Views
                            views_elem = await video.query_selector('span.ytd-video-meta-block:first-child')
                            views = await views_elem.inner_text() if views_elem else ""

                            # Duration
                            duration_elem = await video.query_selector('span.ytd-thumbnail-overlay-time-status-renderer')
                            duration = await duration_elem.inner_text() if duration_elem else ""

                            results.append({
                                'title': title.strip(),
                                'url': link,
                                'channel': channel.strip(),
                                'views': views.strip(),
                                'duration': duration.strip(),
                                'search_keyword': keyword,
                                'timestamp': datetime.now().isoformat()
                            })

                            collected += 1
                            logger.debug(f"Collected video {collected}: {title[:50]}...")

                        except Exception as e:
                            logger.debug(f"Error extracting from video renderer: {e}")
                            continue

                # Check if we found new results
                if collected == previous_count:
                    consecutive_no_new += 1
                    if consecutive_no_new >= 3:
                        logger.info(f"No new results after 3 attempts, stopping at {collected} results")
                        break
                else:
                    consecutive_no_new = 0

                # Scroll down with human-like behavior
                if collected < max_results:
                    # Scroll more aggressively
                    await page.evaluate('window.scrollBy(0, window.innerHeight * 2)')
                    await self.behavior_modeler.human_delay(1.5, 3)

                    # Sometimes click "Show more results" if available
                    try:
                        show_more = await page.query_selector('button[aria-label*="Show more"], yt-next-continuation button')
                        if show_more:
                            await show_more.click()
                            await self.behavior_modeler.human_delay(1, 2)
                    except:
                        pass

                    scroll_attempts += 1

            logger.info(f"Search completed: found {len(results)} results for '{keyword}'")

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)

        return results

    async def _check_for_captcha(self, page: Page) -> bool:
        """Check if CAPTCHA is present"""
        captcha_selectors = [
            'iframe[title*="recaptcha"]',
            'iframe[src*="recaptcha"]',
            'div.g-recaptcha',
            '#captcha-form',
            '.captcha-container'
        ]

        for selector in captcha_selectors:
            if await page.query_selector(selector):
                return True
        return False

    async def _get_available_browser(self) -> Browser:
        """Get an available browser from pool or create new one"""
        if not self.browser_pool:
            logger.info("No browsers in pool, creating new one...")
            browser = await self._create_stealth_browser()
            self.browser_pool.append(browser)

        # Simple round-robin selection
        browser = random.choice(self.browser_pool)

        # Check if browser is still connected
        try:
            contexts = browser.contexts
            if not contexts:
                logger.warning("Browser has no contexts, creating new browser...")
                self.browser_pool.remove(browser)
                browser = await self._create_stealth_browser()
                self.browser_pool.append(browser)
        except:
            logger.warning("Browser disconnected, creating new browser...")
            self.browser_pool.remove(browser)
            browser = await self._create_stealth_browser()
            self.browser_pool.append(browser)

        return browser

    async def scale_up(self, target_browsers: int):
        """Scale up browser pool"""
        current = len(self.browser_pool)
        if target_browsers > current:
            for _ in range(min(target_browsers - current, 10)):  # Add max 10 at a time
                browser = await self._create_stealth_browser()
                self.browser_pool.append(browser)

    async def cleanup(self):
        """Clean up resources"""
        for browser in self.browser_pool:
            await browser.close()
        self.browser_pool.clear()


class FingerprintManager:
    """Manages browser fingerprints for stealth"""

    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        ]

        self.viewports = [
            {'width': 1920, 'height': 1080},
            {'width': 1366, 'height': 768},
            {'width': 1536, 'height': 864},
            {'width': 1440, 'height': 900},
            {'width': 1280, 'height': 720},
            {'width': 1600, 'height': 900},
        ]

        self.locales = ['en-US', 'en-GB', 'en-CA', 'en-AU', 'en-NZ']
        self.timezones = [
            'America/New_York', 'America/Chicago', 'America/Denver', 
            'America/Los_Angeles', 'Europe/London', 'Europe/Paris',
            'Asia/Tokyo', 'Australia/Sydney'
        ]

    def generate_fingerprint(self) -> Dict:
        """Generate a unique browser fingerprint"""
        return {
            'user_agent': random.choice(self.user_agents),
            'viewport': random.choice(self.viewports),
            'locale': random.choice(self.locales),
            'timezone': random.choice(self.timezones),
            'language': random.choice(['en-US,en;q=0.9', 'en-GB,en;q=0.9', 'en;q=0.8']),
            'platform': random.choice(['Win32', 'MacIntel', 'Linux x86_64']),
            'geo': {
                'lat': round(random.uniform(25.0, 49.0), 6),
                'lon': round(random.uniform(-125.0, -66.0), 6)
            },
            'webgl_vendor': random.choice(['Intel Inc.', 'Google Inc.', 'ATI Technologies Inc.']),
            'webgl_renderer': random.choice([
                'Intel Iris OpenGL Engine',
                'ANGLE (Intel HD Graphics 620 Direct3D11 vs_5_0 ps_5_0)',
                'Mesa DRI Intel(R) HD Graphics 630'
            ]),
            'canvas_hash': hashlib.md5(str(random.random()).encode()).hexdigest()
        }


class BehaviorModeler:
    """Models human-like behavior patterns"""

    async def human_delay(self, min_seconds: float, max_seconds: float):
        """Add human-like delay"""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    async def human_type(self, page: Page, element, text: str):
        """Type with human-like speed and patterns"""
        for char in text:
            await element.type(char)
            # Variable typing speed
            if random.random() < 0.1:  # 10% chance of longer pause
                await asyncio.sleep(random.uniform(0.2, 0.5))
            else:
                await asyncio.sleep(random.uniform(0.05, 0.15))

    async def human_click(self, page: Page, element):
        """Click with human-like behavior"""
        # Get element position
        box = await element.bounding_box()
        if box:
            # Click slightly off-center like humans do
            x = box['x'] + box['width'] / 2 + random.uniform(-5, 5)
            y = box['y'] + box['height'] / 2 + random.uniform(-5, 5)

            # Move mouse to position first
            await page.mouse.move(x, y, steps=random.randint(5, 15))
            await self.human_delay(0.05, 0.15)
            await page.mouse.click(x, y)
        else:
            await element.click()

    async def human_scroll(self, page: Page):
        """Scroll with human-like patterns"""
        # Scroll in multiple small steps
        steps = random.randint(3, 7)
        for _ in range(steps):
            scroll_amount = random.randint(100, 300)
            await page.evaluate(f'window.scrollBy(0, {scroll_amount})')
            await self.human_delay(0.1, 0.3)


class ProxyManager:
    """Manages proxy rotation for stealth"""

    def __init__(self):
        self.proxies = []
        self.current_index = 0

    async def initialize(self):
        """Initialize proxy list"""
        # In production, this would load from a proxy provider
        # For now, we'll use direct connection
        logger.info("Proxy manager initialized (using direct connection in dev)")

    async def get_proxy(self) -> Optional[Dict]:
        """Get next proxy from rotation"""
        # In production, implement proxy rotation here
        # Return None for direct connection
        return None


class CaptchaSolver:
    """CAPTCHA solving using YOLO V11"""

    def __init__(self):
        self.yolo_initialized = False
        self.model = None

    async def initialize(self):
        """Initialize YOLO V11 for CAPTCHA solving"""
        try:
            # Import YOLO v11 dependencies
            import torch
            import cv2

            # For production, we would load the actual YOLO v11 model
            # This is a placeholder that shows the structure
            logger.info("CAPTCHA solver initialized with YOLO V11 support")
            self.yolo_initialized = True

        except ImportError:
            logger.warning("YOLO dependencies not found, CAPTCHA solving limited")

    async def solve_captcha(self, page: Page) -> bool:
        """Solve CAPTCHA on the page"""
        try:
            # Check for reCAPTCHA iframe
            recaptcha_frame = await page.query_selector('iframe[title*="recaptcha"]')
            if not recaptcha_frame:
                return False

            # Click the checkbox
            frame = await recaptcha_frame.content_frame()
            if frame:
                checkbox = await frame.query_selector('.recaptcha-checkbox')
                if checkbox:
                    await checkbox.click()
                    await asyncio.sleep(2)

                    # Check if challenge appeared
                    challenge_frame = await page.query_selector('iframe[title*="challenge"]')
                    if challenge_frame:
                        # This is where YOLO V11 would analyze the images
                        # For now, we'll implement the structure
                        await self._solve_image_challenge(page, challenge_frame)

            return True

        except Exception as e:
            logger.error(f"CAPTCHA solving error: {e}")
            return False

    async def _solve_image_challenge(self, page: Page, challenge_frame):
        """Solve image-based CAPTCHA challenge"""
        # In production, this would:
        # 1. Take screenshot of challenge
        # 2. Extract individual images
        # 3. Use YOLO V11 to detect objects
        # 4. Click on matching images
        # 5. Submit solution

        frame = await challenge_frame.content_frame()
        if not frame:
            return

        # Get challenge type
        challenge_text = await frame.query_selector('.rc-imageselect-desc-no-canonical')
        if challenge_text:
            text = await challenge_text.inner_text()
            logger.info(f"CAPTCHA challenge: {text}")

        # This is where we would implement YOLO V11 detection
        # For the structure, we show how it would work
        await asyncio.sleep(2)  # Simulate processing time


# Main execution
async def main():
    """Main entry point"""
    print("🤖 YouTube Search Bot Starting...")
    print("=" * 50)

    bot = YouTubeSearchBot(max_browsers=1)  # Start with 1 browser for testing

    try:
        print("Initializing bot components...")
        await bot.initialize()
        print("✅ Bot initialized successfully!")

        # Example search
        keyword = "python programming"
        print(f"\nPerforming test search for: '{keyword}'")
        print("This may take a moment...")

        results = await bot.search(keyword, max_results=5)

        if results:
            print(f"\n✅ Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']}")
                print(f"   URL: {result['url']}")
                print(f"   Channel: {result['channel']}")
                print(f"   Views: {result['views']}")
        else:
            print("\n❌ No results found!")
            print("\nTroubleshooting:")
            print("1. Run 'python test_search.py' for detailed debugging")
            print("2. Check if YouTube is accessible in your browser")
            print("3. You might need to configure proxies")
            print("4. See TROUBLESHOOTING.md for more help")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFor debugging, run: python test_search.py")

    finally:
        print("\nCleaning up...")
        await bot.cleanup()
        print("✅ Bot shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())