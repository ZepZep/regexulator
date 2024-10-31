# Regexulator: regex from examples using LLMs
Uses Ollama to generate regular expression patterns from examples.

## Usage 
```python
from regexulator import Regexulator

examples = [{
  "string": "It will start at 10:30 and end at 11:21.",
  "match": [
    {"start": 17, "end": 22},
    {"start": 34, "end": 39},
  ]
}]

reg = Regexulator(model="llama3.1:70b", time_limit=10)
pattern = reg.fit(examples)
```
