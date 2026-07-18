# Task 5 Cross-Scenario Smoke Run

This directory verifies the cross-scenario benchmark pipeline.

```text
train scenarios: easy, medium
eval scenarios:  easy, hard
methods:         greedy_damage, maskable_ppo
train seeds:     0, 1
training budget: 16 steps
```

The run is an engineering acceptance test, not a performance experiment. See
`experiment_config.json` for the exact command, package versions, scenario
snapshots, seeds, model settings, and artifact paths.
