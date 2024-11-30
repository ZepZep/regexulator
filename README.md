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


## Configuration
You can use the `RegexulatorConfig` to make a reusable config with type hints. Keyword arguments supplied directly to `Regexulator` overwrite individual config attributes.
```python
from regexulator import Regexulator, RegexulatorConfig

cfg = RegexulatorConfig(
    model="llama3.1:70b",
    time_limit=60,
    initial_splits=4,
    max_depth=5,
    branching_factor=1
)
reg1 = Regexulator(cfg, time_limit=10)
reg2 = Regexulator(cfg, time_limit=200, max_depth=8)
```

## .dot visualisation
You can create a graphviz `.dot`  representation of the exploration tree with the `Regexulator.to_dot()` method.

In a Jupyter notebook, you can just call `Regexulator.show_dot()` (this needs the `graphviz` module)