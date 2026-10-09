## Evaluation results - REAL provider (openai_compatible)

Provider: `openai_compatible` / model `openai/gpt-oss-120b` - 5 products

| Metric | Single-prompt baseline | Agent loop |
|---|---|---|
| Pass rate | 100.0% (5/5) | 40.0% (2/5) |
| Passed on first draft | 100.0% | 0.0% |
| Avg iterations (draft+evaluate cycles) | 1.00 | 1.80 |
| Avg LLM calls per product | 2.0 | 4.8 |
| Avg total tokens | 4375 | 10183 |
| Avg wall-clock latency | 33557.4 ms | 76023.3 ms |
| p95 wall-clock latency | 59492.1 ms | 134340.0 ms |
| Run statuses | {'passed': 5} | {'error': 2, 'passed': 2, 'failed': 1} |

### Most common failure reasons (final evaluation of runs that did not pass)

| Check | Baseline | Agent |
|---|---|---|
| `error` | 0 | 2 |
| `numbers_supported` | 0 | 1 |

### Pass rate by case type

| Case type | Baseline | Agent |
|---|---|---|
| normal | 5/5 | 2/5 |

> Real-provider run on 2026-10-08; n=5, single run, temperature as configured in the provider: expect run-to-run variance.
