## Evaluation results - REAL provider (openai_compatible)

Provider: `openai_compatible` / model `openai/gpt-oss-120b` - 10 products

| Metric | Single-prompt baseline | Agent loop |
|---|---|---|
| Pass rate | 40.0% (4/10) | 0.0% (0/10) |
| Passed on first draft | 40.0% | 0.0% |
| Avg iterations (draft+evaluate cycles) | 0.90 | 0.00 |
| Avg LLM calls per product | 1.5 | 0.5 |
| Avg total tokens | 3204 | 497 |
| Avg wall-clock latency | 18531.2 ms | 69901.3 ms |
| p95 wall-clock latency | 32764.8 ms | 95066.4 ms |
| Run statuses | {'passed': 4, 'failed': 5, 'error': 1} | {'error': 10} |

### Most common failure reasons (final evaluation of runs that did not pass)

| Check | Baseline | Agent |
|---|---|---|
| `error` | 1 | 10 |
| `llm_judge` | 2 | 0 |
| `conflicts_not_used` | 1 | 0 |
| `length` | 1 | 0 |
| `mentions_product` | 1 | 0 |

### Pass rate by case type

| Case type | Baseline | Agent |
|---|---|---|
| conflict | 1/2 | 0/2 |
| dutch_input | 1/1 | 0/1 |
| injection | 1/3 | 0/3 |
| long_specs | 0/1 | 0/1 |
| missing_data | 0/2 | 0/2 |
| unicode | 1/1 | 0/1 |

### Pipeline signals (deterministic, independent of the model)

| Signal | Flagged / expected |
|---|---|
| conflict | 2/2 |
| injection | 3/3 |
| missing_price | 1/1 |
| missing_specs | 2/2 |

> Real-provider run on 2026-10-08; n=10, single run, temperature as configured in the provider: expect run-to-run variance.

> NOTE: the agent column above is INVALID. All 10 agent runs ended in HTTP 429 (free daily token budget used up), so the agent was not measured. The baseline column is valid.
