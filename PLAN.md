# PLAN

Goal: a portfolio-quality "Product Marketing Agent" (FastAPI + CLI + plain HTML page) that turns
structured product data into English and Dutch marketing copy through a plan -> draft ->
evaluate -> revise agent loop. Everything runs offline against a deterministic `MockProvider`;
real providers are implemented but only exercised against local fakes.

## Environment findings (checked before starting)

| Item | Finding |
|---|---|
| Python | 3.13 (CI will use 3.11/3.12 as well, NOT run here) |
| PyPI | reachable through the proxy (packages installable) |
| LLM APIs / Docker Hub | not reachable -> real providers cannot be verified |
| Docker | CLI present, **daemon not running** -> `docker build` cannot be verified here |
| GPU / local LLM | none, not attempted |

## Phases (one local commit each)

1. **Scaffold** - pyproject, ruff, git hygiene, PLAN/PROGRESS/DECISIONS.
2. **Core** - settings, schemas, provider interface + Mock/OpenAI-compatible/Anthropic providers,
   reliability (timeouts, retries/backoff, JSON validation + one repair, budgets), tracing,
   structured logging, versioned prompt files.
3. **Knowledge + tools** - brand-guideline markdown, BM25 (+ optional embeddings), four tools,
   prompt-injection scanner/sanitiser.
4. **Evaluator + agent** - deterministic checks + LLM judge, plan/draft/evaluate/revise loop,
   single-prompt baseline, "why this passed" report.
5. **Interfaces** - FastAPI service, CLI, minimal web page.
6. **Evaluation harness** - 30-product CSV, baseline vs agent comparison with scripted mock,
   script for running the same harness against a real provider.
7. **Tests** - pytest suite (agent loop, tools, evaluator, RAG, providers over a local fake server,
   API, security).
8. **Packaging + docs** - Dockerfile, compose, `.env.example`, GitHub Actions, README, `docs/`
   sample trace and demo output.
9. **Verification + delivery** - ruff, pytest, demo run, zip, report.

## Non-goals

No LangChain/LlamaIndex, no paid APIs, no pushing to any git remote, no invented metrics.
