# Environments

Environment implementations belong here.

Suggested layout:

```text
envs/
  discrete/
  continuous/
  multi_agent/
  air_defense/
```

Environment code should expose Gymnasium or PettingZoo-compatible interfaces and avoid importing agents or algorithms.

## AirDefense v1.0 scenarios

Named v1.0 scenario configurations are defined in:

```text
air_defense_v1/scenarios.py
```

Use the public factory instead of editing the frozen default configuration:

```python
from rein_learning.envs import get_air_defense_v1_scenario

config = get_air_defense_v1_scenario("medium")
```

Canonical profiles include `easy`, `medium`, `hard`, and the single-axis pressure scenarios listed by `list_air_defense_v1_scenarios()`.
