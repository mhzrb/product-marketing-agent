# PROGRESS

## Done
- Phase 1: scaffold, PLAN.md, DECISIONS.md
- Phase 2: providers (mock/simulated, OpenAI-compatible, Anthropic), RAG (BM25 + optional embeddings),
  tools, facts, security, evaluator, agent loop, reliability, tracing, versioned prompts
- Phase 3: FastAPI service + web page, CLI, 30-product dataset, eval harness, real-provider script
- Phase 4: pytest suite (299 tests, 96% coverage), fake OpenAI-compatible server, ruff clean
- Phase 5: Dockerfile, docker-compose, .env.example, GitHub Actions, docs/ (demo, sample trace,
  eval results, injection example), README

## Next (needs things this workspace did not have)
1. Run `evals/run_real_provider.py` against Ollama or a hosted free tier and record the numbers.
2. Run `docker build` / `docker compose up` and let the CI workflow run on GitHub.
3. Try the web page in a real browser.
4. Tune banned-phrase and injection patterns on real model output.

## Known issues / limitations
See "Honest limitations" and the "Verified here / NOT VERIFIED" table in README.md.
