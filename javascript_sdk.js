/**
 * YouTube Search Bot JavaScript/TypeScript SDK
 * A powerful client for the YouTube Search Bot API
 */

// TypeScript interfaces (can be used in .d.ts file)
interface SearchRequest {
  keyword: string;
  maxResults?: number;
  sessionId?: string;
  options?: Record<string, any>;
}

interface SearchResult {
  title: string;
  url: string;
  channel: string;
  views: string;
  duration: string;
  searchKeyword: string;
  timestamp: string;
}

interface SearchResponse {
  searchId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'timeout';
  results?: SearchResult[];
  error?: string;
  timestamp: string;
  durationMs?: number;
}

interface SystemStatus {
  status: string;
  activeSearches: number;
  totalBrowsers: number;
  cpuUsage: number;
  memoryUsage: number;
  uptimeSeconds: number;
  totalSearchesCompleted: number;
  averageSearchTimeMs: number;
  successRate: number;
}

interface BatchSearchResponse {
  searchIds: (string | null)[];
}

// Custom errors
class YouTubeSearchBotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'YouTubeSearchBotError';
  }
}

class RateLimitError extends YouTubeSearchBotError {
  constructor(message: string = 'Rate limit exceeded') {
    super(message);
    this.name = 'RateLimitError';
  }
}

class SearchTimeoutError extends YouTubeSearchBotError {
  constructor(message: string = 'Search timed out') {
    super(message);
    this.name = 'SearchTimeoutError';
  }
}

/**
 * YouTube Search Bot Client
 * 
 * @example
 * const client = new YouTubeSearchBotClient('http://localhost:8000');
 * const results = await client.search('python tutorials', { maxResults: 10 });
 */
class YouTubeSearchBotClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;
  private maxRetries: number;
  private headers: Record<string, string>;

  constructor(baseUrl: string, options?: {
    apiKey?: string;
    timeout?: number;
    maxRetries?: number;
  }) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = options?.apiKey;
    this.timeout = options?.timeout || 300000; // 5 minutes default
    this.maxRetries = options?.maxRetries || 3;

    this.headers = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      this.headers['Authorization'] = `Bearer ${this.apiKey}`;
    }
  }

  /**
   * Make HTTP request with retry logic
   */
  private async request<T>(
    method: string,
    endpoint: string,
    options?: {
      body?: any;
      params?: Record<string, any>;
      retries?: number;
    }
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}${endpoint}`);

    // Add query parameters
    if (options?.params) {
      Object.entries(options.params).forEach(([key, value]) => {
        url.searchParams.append(key, String(value));
      });
    }

    const fetchOptions: RequestInit = {
      method,
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    };

    if (options?.body) {
      fetchOptions.body = JSON.stringify(options.body);
    }

    const retries = options?.retries ?? this.maxRetries;

    try {
      const response = await fetch(url.toString(), fetchOptions);
      const data = await response.json();

      if (response.status === 429) {
        throw new RateLimitError();
      }

      if (!response.ok) {
        const errorMessage = data.detail || `HTTP ${response.status}`;
        throw new YouTubeSearchBotError(`API error: ${errorMessage}`);
      }

      return data;
    } catch (error) {
      if (retries > 0 && !(error instanceof RateLimitError)) {
        // Exponential backoff
        const delay = Math.pow(2, this.maxRetries - retries) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));

        return this.request<T>(method, endpoint, {
          ...options,
          retries: retries - 1,
        });
      }

      throw error;
    }
  }

  /**
   * Perform a YouTube search
   * 
   * @example
   * const results = await client.search('javascript tutorials', {
   *   maxResults: 20,
   *   waitForResults: true
   * });
   */
  async search(
    keyword: string,
    options?: {
      maxResults?: number;
      sessionId?: string;
      searchOptions?: Record<string, any>;
      waitForResults?: boolean;
      pollingInterval?: number;
    }
  ): Promise<SearchResult[]> {
    const request: SearchRequest = {
      keyword,
      maxResults: options?.maxResults || 50,
      sessionId: options?.sessionId,
      options: options?.searchOptions,
    };

    // Submit search
    const response = await this.request<SearchResponse>('/search', {
      method: 'POST',
      body: request,
    });

    if (!options?.waitForResults ?? true) {
      return [];
    }

    // Poll for results
    const startTime = Date.now();
    const maxWait = this.timeout;
    const pollingInterval = options?.pollingInterval || 1000;

    while (Date.now() - startTime < maxWait) {
      const status = await this.getSearchResults(response.searchId);

      if (status.status === 'completed') {
        return status.results || [];
      } else if (status.status === 'failed') {
        throw new YouTubeSearchBotError(`Search failed: ${status.error}`);
      } else if (status.status === 'timeout') {
        throw new SearchTimeoutError();
      }

      await new Promise(resolve => setTimeout(resolve, pollingInterval));
    }

    throw new SearchTimeoutError(`Search timed out after ${maxWait}ms`);
  }

  /**
   * Submit a search without waiting for results
   */
  async searchAsync(
    keyword: string,
    options?: {
      maxResults?: number;
      sessionId?: string;
      searchOptions?: Record<string, any>;
    }
  ): Promise<string> {
    const request: SearchRequest = {
      keyword,
      maxResults: options?.maxResults || 50,
      sessionId: options?.sessionId,
      options: options?.searchOptions,
    };

    const response = await this.request<SearchResponse>('/search', {
      method: 'POST',
      body: request,
    });

    return response.searchId;
  }

  /**
   * Get results for a previously submitted search
   */
  async getSearchResults(searchId: string): Promise<SearchResponse> {
    return this.request<SearchResponse>(`/search/${searchId}`, {
      method: 'GET',
    });
  }

  /**
   * Perform multiple searches in batch
   * 
   * @example
   * const results = await client.batchSearch(['python', 'javascript', 'golang']);
   */
  async batchSearch(
    keywords: string[],
    options?: {
      maxResults?: number;
      sessionId?: string;
    }
  ): Promise<Record<string, SearchResult[]>> {
    const requests: SearchRequest[] = keywords.map(keyword => ({
      keyword,
      maxResults: options?.maxResults || 50,
      sessionId: options?.sessionId,
    }));

    const response = await this.request<BatchSearchResponse>('/search/batch', {
      method: 'POST',
      body: requests,
    });

    const results: Record<string, SearchResult[]> = {};

    // Wait for all searches to complete
    await Promise.all(
      keywords.map(async (keyword, index) => {
        const searchId = response.searchIds[index];
        if (searchId) {
          try {
            const searchResults = await this.search(keyword, {
              maxResults: options?.maxResults,
              sessionId: options?.sessionId,
              waitForResults: true,
            });
            results[keyword] = searchResults;
          } catch (error) {
            console.error(`Batch search failed for '${keyword}':`, error);
            results[keyword] = [];
          }
        } else {
          results[keyword] = [];
        }
      })
    );

    return results;
  }

  /**
   * Stream search results using Server-Sent Events
   * 
   * @example
   * for await (const update of client.streamSearchResults(searchId)) {
   *   console.log(`Status: ${update.status}`);
   *   if (update.results) {
   *     console.log(`Found ${update.results.length} results`);
   *   }
   * }
   */
  async *streamSearchResults(searchId: string): AsyncIterator<SearchResponse> {
    const url = `${this.baseUrl}/search/${searchId}/stream`;
    const response = await fetch(url, {
      headers: this.headers,
    });

    if (!response.ok) {
      throw new YouTubeSearchBotError(`Failed to start stream: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new YouTubeSearchBotError('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            yield data as SearchResponse;

            if (['completed', 'failed', 'timeout'].includes(data.status)) {
              return;
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Get system status
   */
  async getStatus(): Promise<SystemStatus> {
    return this.request<SystemStatus>('/status', {
      method: 'GET',
    });
  }

  /**
   * Check API health
   */
  async checkHealth(): Promise<boolean> {
    try {
      const response = await this.request<{ status: string }>('/health', {
        method: 'GET',
      });
      return response.status === 'healthy';
    } catch {
      return false;
    }
  }

  /**
   * Get detailed system metrics
   */
  async getMetrics(): Promise<Record<string, any>> {
    return this.request<Record<string, any>>('/metrics', {
      method: 'GET',
    });
  }

  /**
   * Manually scale the system
   */
  async scale(action: 'up' | 'down', count: number = 1): Promise<{ message: string }> {
    return this.request<{ message: string }>('/admin/scale', {
      method: 'POST',
      params: { action, count },
    });
  }
}

/**
 * Simplified interface for common use cases
 */
class YouTubeSearchBot {
  private client: YouTubeSearchBotClient;

  constructor(baseUrl: string, apiKey?: string) {
    this.client = new YouTubeSearchBotClient(baseUrl, { apiKey });
  }

  /**
   * Simple search interface
   */
  async search(keyword: string, maxResults: number = 50): Promise<SearchResult[]> {
    return this.client.search(keyword, { maxResults });
  }

  /**
   * Batch search interface
   */
  async batchSearch(
    keywords: string[],
    maxResults: number = 50
  ): Promise<Record<string, SearchResult[]>> {
    return this.client.batchSearch(keywords, { maxResults });
  }

  /**
   * Get system status
   */
  async getStatus(): Promise<SystemStatus> {
    return this.client.getStatus();
  }
}

// Node.js style exports
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    YouTubeSearchBotClient,
    YouTubeSearchBot,
    YouTubeSearchBotError,
    RateLimitError,
    SearchTimeoutError,
  };
}

// ES6 exports
export {
  YouTubeSearchBotClient,
  YouTubeSearchBot,
  YouTubeSearchBotError,
  RateLimitError,
  SearchTimeoutError,
};

// TypeScript type exports
export type {
  SearchRequest,
  SearchResult,
  SearchResponse,
  SystemStatus,
  BatchSearchResponse,
};

// Example usage
/*
// Basic usage
const bot = new YouTubeSearchBot('http://localhost:8000');
const results = await bot.search('javascript tutorials', 10);

results.forEach(result => {
  console.log(`${result.title} - ${result.url}`);
});

// Advanced usage with client
const client = new YouTubeSearchBotClient('http://localhost:8000', {
  apiKey: 'your-api-key',
  timeout: 60000,
  maxRetries: 5
});

// Stream results
const searchId = await client.searchAsync('python programming');
for await (const update of client.streamSearchResults(searchId)) {
  console.log(`Search status: ${update.status}`);
  if (update.results) {
    console.log(`Found ${update.results.length} results so far`);
  }
}

// Batch search
const batchResults = await client.batchSearch([
  'react tutorial',
  'vue.js guide',
  'angular basics'
], { maxResults: 5 });

Object.entries(batchResults).forEach(([keyword, results]) => {
  console.log(`\nResults for "${keyword}":`);
  results.forEach((result, i) => {
    console.log(`${i + 1}. ${result.title}`);
  });
});

// System monitoring
const status = await client.getStatus();
console.log(`System Status: ${status.status}`);
console.log(`Active Searches: ${status.activeSearches}`);
console.log(`Success Rate: ${(status.successRate * 100).toFixed(2)}%`);
*/