
from flask import Flask, render_template, request, jsonify, session
import requests
import json
import asyncio
import aiohttp
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = 'youtube-search-bot-secret-key'

# API server configuration
API_BASE_URL = 'http://localhost:8000'

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
        })
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Search failed'}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<search_id>')
def get_search_results(search_id):
    try:
        response = requests.get(f'{API_BASE_URL}/search/{search_id}')
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Search not found'}), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    try:
        response = requests.get(f'{API_BASE_URL}/status')
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': 'Status unavailable'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
