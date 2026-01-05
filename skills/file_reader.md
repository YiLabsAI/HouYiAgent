# file_reader

## Description
Read and return the contents of a text file.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "filepath": {
      "type": "string",
      "description": "Path to the file to read"
    },
    "encoding": {
      "type": "string",
      "description": "File encoding",
      "default": "utf-8"
    }
  },
  "required": ["filepath"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "content": {
      "type": "string",
      "description": "File contents"
    },
    "filepath": {
      "type": "string",
      "description": "Path that was read"
    },
    "size": {
      "type": "integer",
      "description": "File size in bytes"
    }
  }
}
```

## Examples

### Example 1: Read Text File
**Input:**
```json
{"filepath": "data/sample.txt"}
```

**Output:**
```json
{
  "content": "Hello, World!\nThis is a sample file.",
  "filepath": "data/sample.txt",
  "size": 42
}
```

## Implementation
- Language: Python
- Runtime: sync
- Timeout: 10s
- Cost: $0.0001 per call

## Dependencies
- None (uses built-in open)

## Constraints
- Max file size: 10MB
- Text files only
- Read-only access
