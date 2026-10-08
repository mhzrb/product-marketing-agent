# DECISIONS

Short architecture decision log. Newest at the bottom.

## D1 - No agent framework
The agent loop, tool runtime, retrieval and evaluator are written directly on `httpx` + `pydantic`.
The loop is ~200 lines and every step is visible in the trace; a framework would hide exactly the
behaviour this project is meant to demonstrate.

## D2 - Synchronous core
The agent is a sequential pipeline (each step needs the previous output). FastAPI endpoints are
plain `def`, so Starlette runs them in its thread pool. Simpler to test and reason about than async.

## D3 - Mock first, with a *simulated* model
`MockProvider` has three modes: a scripted queue (unit tests), a handler callable, and a built-in
`SimulatedMarketingLLM` that writes template copy and can inject *scripted flaws* per attempt
(unsupported number, banned claim, too long, ...). The flaws are authored by us, so evaluation
numbers from the mock describe the pipeline, not model quality. This is stated in the README.

## D4 - Pipeline control flow is code, not the model
The model never decides whether a draft passes. Pass/fail is computed by the evaluator
(deterministic checks, then an LLM judge that can only *add* failures). The "why this passed"
report is assembled from evaluator output, not from the model's description of its own work.

## D5 - Product data is sanitised before the model sees it
Instruction-like sentences (EN + NL heuristics) are removed from every free-text field and
reported in `security`. This is deliberately *in addition to* output verification (injection
phrases / payload echo / unsupported numbers are rejected by the evaluator). Trade-off: regex
heuristics can false-positive; flagged text is reported, never silently dropped.

## D6 - "Every number must exist in the data" is the core grounding check
Numbers are normalised (`1.299,00` == `1,299.00` == `1299`) and unit pairs (`16 GB`, `27 inch`)
must match as pairs. It is strict on purpose: unsupported figures are the most damaging error in
B2B copy. Known cost: legitimate figures that are not in the data (e.g. "24/7") are rejected.

## D7 - Conflicting data is excluded, not resolved
If the same attribute appears twice with different values (also across EN/NL key synonyms such
as RAM/Memory/Geheugen), neither value may be used; the conflict is reported for a human. Guessing
which value is right would be an invented fact.

## D8 - Two tool transports, one runtime
Native tool calling when the provider supports it, otherwise a validated JSON protocol
(`{"action":"tool"|"final"}`), each reply validated with Pydantic plus one repair attempt. Both
paths are covered by tests and produce identical results with the simulated model.

## D9 - One LLM choke point (`LLMClient`)
Retries with equal-jitter exponential backoff (honouring `Retry-After`), budget checks before every
call (tokens, cost, call count), tracing and token accounting live in one place, so no code path
can skip them.

## D10 - Fail closed
An unusable judge verdict fails the iteration; budget exhaustion returns `budget_exceeded` with the
last unapproved draft marked as not verified; invalid model JSON gets exactly one repair attempt.

## D11 - Fresh context per draft/revise call
Each revise call re-sends product data, plan, guidelines, the previous draft and the feedback
instead of growing a chat history. Costs a few more prompt tokens but keeps calls independent,
cacheable and easy to trace.

## D12 - Own BM25, optional embeddings
BM25 over ~30 guideline chunks is enough and has no dependency. Embeddings are optional and
behind `RETRIEVER=embeddings|hybrid` with RRF fusion. The offline `HashingEmbedder` is lexical
(not semantic) and exists to test the code path; a real embedding endpoint is NOT VERIFIED.

## D13 - Demo/eval numbers are scripted
Flaws are scripted per product in `evals/products.csv` (`mock_script`). The README says so next
to every table; `evals/run_real_provider.py` is the path to real numbers.
