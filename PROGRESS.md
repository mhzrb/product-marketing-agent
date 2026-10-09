# PROGRESS

## Done
- Phase 1: scaffold, PLAN.md, DECISIONS.md
- Phase 2: providers (mock/simulated, OpenAI-compatible, Anthropic), RAG (BM25 + optional embeddings),
  tools, facts, security, evaluator, agent loop, reliability, tracing, versioned prompts
- Phase 3: FastAPI service + web page, CLI, 30-product dataset, eval harness, real-provider script
- Phase 4: pytest suite (312 tests, 96% coverage), fake OpenAI-compatible server, ruff clean
- Phase 5: Dockerfile, docker-compose, .env.example, GitHub Actions, docs/ (demo, sample trace,
  eval results, injection example), README

## Next (needs things this workspace did not have)
1. Done in small form (Groq free tier, see README "Real-model run"). Still open: agent vs baseline on
   `evals/products_hard10.csv` (blocked by the free daily token budget), and Ollama.
2. Done: CI (tests on 3.11-3.13, ruff, Docker build and smoke test) is green on GitHub. Still open: `docker compose up`.
3. Done by hand with the mock provider (9 Oct). Still open: the page with a real provider.
4. Tune banned-phrase and injection patterns on real model output.

## Known issues / limitations
See "Honest limitations" and the "Verified here / NOT VERIFIED" table in README.md.
