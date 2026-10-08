## Evaluation results - MOCK provider (scripted simulation, not a real model)

Provider: `mock` / model `mock-1` - 30 products

| Metric | Single-prompt baseline | Agent loop |
|---|---|---|
| Pass rate | 36.7% (11/30) | 93.3% (28/30) |
| Passed on first draft | 36.7% | 36.7% |
| Avg iterations (draft+evaluate cycles) | 1.00 | 1.80 |
| Avg LLM calls per product | 1.5 | 6.9 |
| Avg total tokens (estimated, ~4 chars/token) | 1407 | 7306 |
| Avg wall-clock latency | 2.2 ms | 5.2 ms |
| p95 wall-clock latency | 2.7 ms | 7.7 ms |
| Run statuses | {'passed': 11, 'failed': 19} | {'passed': 28, 'failed': 2} |

### Most common failure reasons (final evaluation of runs that did not pass)

| Check | Baseline | Agent |
|---|---|---|
| `numbers_supported` | 6 | 1 |
| `banned_claims` | 6 | 1 |
| `specs_supported` | 5 | 1 |
| `injection_echo` | 4 | 1 |
| `conflicts_not_used` | 3 | 0 |
| `length` | 2 | 0 |
| `llm_judge` | 2 | 0 |
| `schema_complete` | 2 | 0 |

### Pass rate by case type

| Case type | Baseline | Agent |
|---|---|---|
| conflict | 1/4 | 4/4 |
| dutch_input | 1/4 | 4/4 |
| injection | 2/6 | 5/6 |
| long_specs | 1/1 | 1/1 |
| missing_data | 2/4 | 4/4 |
| normal | 3/7 | 7/7 |
| persistent_fail | 0/1 | 0/1 |
| single_language | 0/1 | 1/1 |
| slow_fix | 0/1 | 1/1 |
| unicode | 1/1 | 1/1 |

### Pipeline signals (deterministic, independent of the model)

| Signal | Flagged / expected |
|---|---|
| conflict | 4/4 |
| injection | 6/6 |
| missing_price | 2/2 |
| missing_specs | 2/2 |

> These numbers come from the simulated mock provider with flaws scripted in `evals/products.csv`. They show that the pipeline detects, reports and repairs the scripted failure types; they say nothing about real model quality.
