# text_analyzer

## Description
Analyze text and return statistics.

## Input Schema
```json
{
  "properties": {
    "text": {
      "title": "Text",
      "type": "string"
    },
    "language": {
      "default": "en",
      "title": "Language",
      "type": "string"
    }
  },
  "required": [
    "text"
  ],
  "title": "Text_AnalyzerInput",
  "type": "object"
}
```

## Output Schema
```json
{
  "properties": {
    "result": {
      "additionalProperties": true,
      "title": "Result",
      "type": "object"
    }
  },
  "required": [
    "result"
  ],
  "title": "Text_AnalyzerOutput",
  "type": "object"
}
```

## Examples

### Example 1
**Input:**
```json
{
  "text": "Hello world",
  "language": "en"
}
```

**Output:**
```json
{
  "word_count": 2,
  "char_count": 11,
  "language": "en"
}
```

## Implementation
- Language: Python
- Runtime: sync
- Timeout: 5s
- Cost: $0.0001 per call

