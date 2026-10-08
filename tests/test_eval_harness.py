from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pma.config import Settings
from pma.dataset import load_cases, parse_specs
from pma.evaluation import render_markdown, run_comparison
from pma.providers.mock import MockProvider

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "evals" / "products.csv"


@pytest.fixture(scope="module")
def cases():
    return load_cases(CSV)


@pytest.fixture(scope="module")
def report(cases, tmp_path_factory):
    settings = replace(Settings(), trace_dir=tmp_path_factory.mktemp("traces"))
    provider = MockProvider(
        flaw_scripts={c.product.id: c.mock_script for c in cases}, record_requests=False
    )
    return run_comparison(cases, provider, settings, sleep=lambda _s: None)


def test_dataset_has_30_products_with_the_required_tricky_cases(cases):
    assert len(cases) == 30 and len({c.product.id for c in cases}) == 30
    types = {c.case_type for c in cases}
    assert {"normal", "dutch_input", "missing_data", "conflict", "injection"} <= types
    assert sum(c.case_type == "injection" for c in cases) >= 5
    assert any(not c.product.specs for c in cases) and any(c.product.price is None for c in cases)
    assert any(c.languages == ["nl"] for c in cases)
    assert all(
        f in {"none", *__import__("pma.providers.simulated", fromlist=["FLAWS"]).FLAWS}
        for c in cases
        for f in c.mock_script
    )


def test_parse_specs_keeps_duplicate_keys_as_separate_entries():
    specs = parse_specs("RAM=16 GB|RAM=32 GB|Note without equals|Storage=1 TB")
    assert specs == {"RAM": "16 GB", "RAM (2)": "32 GB", "Storage": "1 TB"}


def test_every_scripted_case_is_actually_handled_as_scripted(report, cases):
    """The harness is a *simulation*: check it behaves exactly as scripted, so the table in the
    README can be reproduced and the pipeline mechanics are what is being asserted."""
    by_id = {o.id: o for o in report.agent.outcomes}
    baseline = {o.id: o for o in report.baseline.outcomes}
    for case in cases:
        pid = case.product.id
        script = case.mock_script
        clean_first = not script or script == ["invalid_json"]
        assert baseline[pid].passed == clean_first, pid  # baseline sees only the first attempt
        if script and script[-1] != "none" and script != ["invalid_json"]:
            assert not by_id[pid].passed, pid  # never repaired within 3 iterations
        else:
            assert by_id[pid].passed, pid
            assert by_id[pid].iterations == (
                len([s for s in script if s != "none"]) + 1
                if script and script != ["invalid_json"]
                else 1
            ), pid


def test_agent_beats_baseline_on_the_scripted_set_and_costs_more(report):
    b, a = report.baseline, report.agent
    assert (b.passed, a.passed, b.n) == (11, 28, 30)
    assert a.avg_iterations > b.avg_iterations == 1.0
    assert a.avg_llm_calls > b.avg_llm_calls and a.avg_total_tokens > b.avg_total_tokens
    assert a.first_iteration_pass_rate == pytest.approx(b.first_iteration_pass_rate)
    assert set(a.failure_reasons) <= {
        "numbers_supported",
        "specs_supported",
        "banned_claims",
        "injection_echo",
    }
    assert a.statuses == {"passed": 28, "failed": 2}


def test_pipeline_signals_are_detected_independently_of_the_model(report):
    assert report.signal_detection == {
        "conflict": "4/4", "injection": "6/6", "missing_price": "2/2", "missing_specs": "2/2"
    }  # fmt: skip


def test_markdown_report_contains_all_sections(report):
    report.notes.append("note text")
    md = render_markdown(report)
    for needle in ("| Pass rate |", "Avg iterations", "Avg LLM calls", "wall-clock latency",
                   "failure reasons", "Pass rate by case type", "Pipeline signals", "> note text"):  # fmt: skip
        assert needle in md
    data = json.loads(report.to_json())
    assert data["agent"]["passed"] == 28 and len(data["agent"]["outcomes"]) == 30


def test_single_approach_runs(cases):
    settings = replace(Settings(), trace_dir=Path("/tmp/pma-test-traces"))
    provider = MockProvider(record_requests=False)
    only = run_comparison(
        cases[:3], provider, settings, approaches=("agent",), sleep=lambda _s: None
    )
    assert only.baseline is None and only.agent.n == 3
    assert "pass rate" in render_markdown(only)


# ------------------------------------------------------------------ the real-provider script


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "evals" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_provider_script_refuses_to_run_with_the_mock(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert load_script("run_real_provider").main(["--yes"]) == 2
    assert "Refusing to run" in capsys.readouterr().err


def test_real_provider_script_reports_config_errors(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_script("run_real_provider").main(["--yes"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_real_provider_script_runs_end_to_end_against_a_fake_server(monkeypatch, tmp_path, capsys):
    """Exercises the script with LLM_PROVIDER=openai_compatible against a local fake server."""
    from .fake_openai import FakeServer

    with FakeServer() as server:
        monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_MODEL", "fake-model")
        monkeypatch.setenv("LLM_BACKOFF_BASE_S", "0")
        code = load_script("run_real_provider").main(
            ["--yes", "--limit", "2", "--out-dir", str(tmp_path)]
        )
    out = capsys.readouterr().out
    assert code == 0 and "REAL provider" in out and "| Pass rate |" in out
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 2 and files[0].startswith("real_openai_compatible_")
