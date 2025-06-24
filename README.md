
# YouTube Search Bot API

A powerful YouTube search automation system with advanced anti-detection capabilities.

## API Access

### Local Development
- API Server: `http://localhost:8000`
- Web Interface: `http://localhost:5000`

### Production/External Access
After deploying on Replit, your API will be available at:
- `https://your-repl-name.your-username.repl.co`

## Connecting from Other Projects

### Python
```python
from python_sdk import YouTubeSearchBotClient

# For deployed version
client = YouTubeSearchBotClient("https://your-repl-name.your-username.repl.co")

# For local development
# client = YouTubeSearchBotClient("http://localhost:8000")

async with client:
    results = await client.search("python tutorials", max_results=10)
    for result in results:
        print(f"{result.title} - {result.url}")
```

### JavaScript/Node.js
```javascript
import { YouTubeSearchBotClient } from './javascript_sdk.js';

// For deployed version
const client = new YouTubeSearchBotClient('https://your-repl-name.your-username.repl.co');

// For local development  
// const client = new YouTubeSearchBotClient('http://localhost:8000');

const results = await client.search('javascript tutorials', { maxResults: 10 });
results.forEach(result => {
    console.log(`${result.title} - ${result.url}`);
});
```

### Direct HTTP Requests
```bash
# Search request
curl -X POST "https://your-repl-name.your-username.repl.co/search" \
     -H "Content-Type: application/json" \
     -d '{"keyword": "python tutorials", "max_results": 10}'

# Get search results
curl "https://your-repl-name.your-username.repl.co/search/{search_id}"

# Check system status
curl "https://your-repl-name.your-username.repl.co/status"
```

## API Endpoints

- `POST /search` - Submit a search request
- `GET /search/{search_id}` - Get search results
- `POST /search/batch` - Submit multiple searches
- `GET /search/{search_id}/stream` - Stream search results
- `GET /status` - Get system status
- `GET /health` - Health check
- `GET /metrics` - System metrics

## Running the Services

```bash
# Start API server
python api_server.py

# Start web interface (in another terminal)
python web_server.py
```

The API server runs on port 8000 and the web interface on port 5000.
