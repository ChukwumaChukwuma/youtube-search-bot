
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import asyncio
import threading
import time
import json
import logging
from datetime import datetime
import os

# Import your existing bot components
from api_server import BotManager, RateLimiter

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global bot manager (will be initialized in a separate thread)
bot_manager = None
bot_manager_ready = False

def run_bot_manager():
    """Run bot manager in asyncio event loop"""
    global bot_manager, bot_manager_ready
    
    async def init_bot_manager():
        global bot_manager, bot_manager_ready
        from api_server import BotManager
        
        bot_manager = BotManager()
        initial_bots = int(os.getenv('INITIAL_BOTS', '3'))
        await bot_manager.initialize(initial_bots)
        bot_manager_ready = True
        logger.info("Bot manager initialized and ready")
        
        # Keep the event loop running
        while True:
            await asyncio.sleep(1)
    
    # Create new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(init_bot_manager())
    except Exception as e:
        logger.error(f"Bot manager initialization failed: {e}")

# Start bot manager in background thread
bot_thread = threading.Thread(target=run_bot_manager, daemon=True)
bot_thread.start()

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'bot_manager_ready': bot_manager_ready,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """Get system status"""
    if not bot_manager_ready or not bot_manager:
        return jsonify({
            'status': 'initializing',
            'active_searches': 0,
            'total_browsers': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'uptime_seconds': 0.0,
            'total_searches_completed': 0,
            'average_search_time_ms': 0.0,
            'success_rate': 0.0
        })
    
    try:
        status_data = bot_manager.get_system_status()
        return jsonify({
            'status': status_data.status,
            'active_searches': status_data.active_searches,
            'total_browsers': status_data.total_browsers,
            'cpu_usage': status_data.cpu_usage,
            'memory_usage': status_data.memory_usage,
            'uptime_seconds': status_data.uptime_seconds,
            'total_searches_completed': status_data.total_searches_completed,
            'average_search_time_ms': status_data.average_search_time_ms,
            'success_rate': status_data.success_rate
        })
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def submit_search():
    """Submit a search request"""
    if not bot_manager_ready or not bot_manager:
        return jsonify({'error': 'Bot manager not ready'}), 503
    
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        max_results = data.get('max_results', 50)
        
        if not keyword:
            return jsonify({'error': 'Keyword is required'}), 400
        
        if max_results < 1 or max_results > 500:
            return jsonify({'error': 'max_results must be between 1 and 500'}), 400
        
        # Create search request object
        from api_server import SearchRequest
        search_request = SearchRequest(
            keyword=keyword,
            max_results=max_results,
            session_id=request.headers.get('X-Session-ID')
        )
        
        # Submit search using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def submit():
            return await bot_manager.submit_search(search_request)
        
        search_id = loop.run_until_complete(submit())
        loop.close()
        
        return jsonify({
            'search_id': search_id,
            'status': 'queued',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Search submission error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search/<search_id>')
def get_search_results(search_id):
    """Get search results"""
    if not bot_manager_ready or not bot_manager:
        return jsonify({'error': 'Bot manager not ready'}), 503
    
    try:
        # Get search status using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_status():
            return await bot_manager.get_search_status(search_id)
        
        result = loop.run_until_complete(get_status())
        loop.close()
        
        if not result:
            return jsonify({'error': 'Search not found'}), 404
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Get search results error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def get_metrics():
    """Get detailed system metrics"""
    if not bot_manager_ready or not bot_manager:
        return jsonify({'error': 'Bot manager not ready'}), 503
    
    try:
        import psutil
        
        metrics = {
            'bot_metrics': bot_manager.metrics,
            'system': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory': dict(psutil.virtual_memory()._asdict()),
                'disk': dict(psutil.disk_usage('/')._asdict()),
            },
            'queue_size': bot_manager.search_queue.qsize(),
            'active_searches': len(bot_manager.active_searches),
            'total_bots': len(bot_manager.bots)
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Wait a moment for bot manager to initialize
    print("Starting YouTube Search Bot Web Server...")
    print("Waiting for bot manager to initialize...")
    
    # Wait for bot manager to be ready
    timeout = 60  # 60 seconds timeout
    start_time = time.time()
    
    while not bot_manager_ready and (time.time() - start_time) < timeout:
        time.sleep(1)
        print(".", end="", flush=True)
    
    if bot_manager_ready:
        print("\n✅ Bot manager ready!")
        print("🚀 Starting web server on http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    else:
        print("\n❌ Bot manager failed to initialize in time")
        print("Check the logs for errors")
