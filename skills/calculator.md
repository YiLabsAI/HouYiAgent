# calculator

## Description
Perform mathematical calculations on expressions.

## Input Schema
```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
    }
  },
  "required": ["expression"]
}
```

## Output Schema
```json
{
  "type": "object",
  "properties": {
    "result": {
      "type": "number",
      "description": "Result of the calculation"
    },
    "expression": {
      "type": "string",
      "description": "Original expression"
    }
  }
}
```

## Examples

### Example 1: Simple Addition
**Input:**
```json
{"expression": "2 + 2"}
```

**Output:**
```json
{
  "result": 4,
  "expression": "2 + 2"
}
```

### Example 2: Complex Expression
**Input:**
```json
{"expression": "(10 + 5) * 2"}
```

**Output:**
```json
{
  "result": 30,
  "expression": "(10 + 5) * 2"
}
```

## Implementation
- Language: Python
- Runtime: sync
- Timeout: 5s
- Cost: $0.0001 per call

## Dependencies
- None (uses built-in eval)

## Constraints
- Max expression length: 200 characters
- No file system access
- No network access
