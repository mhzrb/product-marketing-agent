## Evaluation results - REAL provider (openai_compatible)

Provider: `openai_compatible` / model `openai/gpt-oss-120b` - 10 products

agent loop: pass rate 50.0% (5/10), avg iterations 1.30, avg LLM calls 5.4

### Pipeline signals (deterministic, independent of the model)

| Signal | Flagged / expected |
|---|---|
| conflict | 2/2 |
| injection | 3/3 |
| missing_price | 1/1 |
| missing_specs | 2/2 |

> Real-provider run on 2026-10-09; n=10, single run, temperature as configured in the provider: expect run-to-run variance.
> Note: 4 of the 10 agent runs in this file died with HTTP 400 tool_use_failed (bug 5 in the README). Fixed afterwards; see the 16 Oct run.
