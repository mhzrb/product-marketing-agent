## Evaluation results - REAL provider (openai_compatible)

Provider: `openai_compatible` / model `openai/gpt-oss-120b` - 10 products

| Metric | Single-prompt baseline | Agent loop |
|---|---|---|
| Pass rate | 10.0% (1/10) | 80.0% (8/10) |
| Passed on first draft | 10.0% | 20.0% |
| Avg iterations (draft+evaluate cycles) | 1.00 | 2.30 |
| Avg LLM calls per product | 1.9 | 7.3 |
| Avg total tokens (estimated, ~4 chars/token) | 4138 | 13712 |
| Avg wall-clock latency | 32507.7 ms | 105308.8 ms |
| p95 wall-clock latency | 87584.1 ms | 144693.3 ms |
| Run statuses | {'failed': 9, 'passed': 1} | {'passed': 8, 'failed': 2} |

### Most common failure reasons (final evaluation of runs that did not pass)

| Check | Baseline | Agent |
|---|---|---|
| `llm_judge` | 6 | 0 |
| `schema_complete` | 2 | 0 |
| `length` | 1 | 1 |
| `conflicts_not_used` | 0 | 1 |
| `mentions_product` | 1 | 0 |

### Pass rate by case type

| Case type | Baseline | Agent |
|---|---|---|
| conflict | 1/2 | 1/2 |
| dutch_input | 0/1 | 1/1 |
| injection | 0/3 | 3/3 |
| long_specs | 0/1 | 1/1 |
| missing_data | 0/2 | 2/2 |
| unicode | 0/1 | 0/1 |

### Pipeline signals (deterministic, independent of the model)

| Signal | Flagged / expected |
|---|---|
| conflict | 2/2 |
| injection | 3/3 |
| missing_price | 1/1 |
| missing_specs | 2/2 |

> Real-provider run on 2026-10-16; n=10, single run, temperature as configured in the provider: expect run-to-run variance.
> Note: 5 of the 10 agent plans were recovered from a rejected tool call named json (bug 5 in the README).
