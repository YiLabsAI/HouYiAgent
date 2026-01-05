# web_search

## Description
Search the web for information using a search engine.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of results",
      "default": 10
    }
  },
  "required": ["query"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "url": {"type": "string"},
          "snippet": {"type": "string"}
        }
      }
    }
  }
}
```

## Examples

### Example 1: Simple Search
**Input:**
```json
{"query": "Python tutorials"}
```

**Output:**
```json
{
  "results": [
    {
      "title": "Python Tutorial - W3Schools",
      "url": "https://www.w3schools.com/python/",
      "snippet": "Learn Python programming with our comprehensive tutorial..."
    }
  ]
}
```

## Implementation
- Language: Python
- Runtime: async
- Timeout: 30s
- Cost: $0.001 per call

## Dependencies
- requests>=2.28.0
- beautifulsoup4>=4.11.0

## Constraints
- Rate limit: 100 calls/minute
- Max query length: 500 characters
