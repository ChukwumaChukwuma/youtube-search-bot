# Updated web server for deployment compatibility
from flask import Flask, render_template, request, jsonify, session
import requests
import json
import asyncio
import aiohttp
from datetime import datetime
import uuid
import os

app = Flask(__name__)
app.secret_key = 'youtube-search-bot-secret-key'

# API server configuration
API_BASE_URL = 'http://0.0.0.0:8000'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        keyword = data.get('keyword', '')
        max_results = data.get('maxResults', 50)

        if not keyword:
            return jsonify({'error': 'Keyword is required'}), 400

        # Submit search to API
        response = requests.post(f'{API_BASE_URL}/search', json={
            'keyword': keyword,
            'max_results': max_results,
            'session_id': session.get('session_id', str(uuid.uuid4()))
        }, timeout=30)

        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Search failed'}), response.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'API server not available. Please start the API server first.'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'API server timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<search_id>')
def get_search_results(search_id):
    try:
        response = requests.get(f'{API_BASE_URL}/search/{search_id}', timeout=30)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Search not found'}), response.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'API server not available'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'API server timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    try:
        response = requests.get(f'{API_BASE_URL}/status', timeout=10)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Status unavailable'}), 500
    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'API server not available. Please start the API server first.'}), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'API server timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    # Get port from environment or default to 5000
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"

    app.run(host="0.0.0.0", port=port, debug=debug_mode)