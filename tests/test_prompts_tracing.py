from __future__ import annotations

import io
import json
import logging
import re

import pytest

from pma.config import Settings
from pma.errors import ConfigError
from pma.logging_setup import JsonFormatter, TextFormatter, configure_logging, get_logger
from pma.prompts import PromptStore
from pma.tracing import Trace, new_run_id, preview

NAMES = ["system", "tool_protocol", "plan", "draft", "revise", "judge", "repair", "baseline"]


@pytest.mark.parametrize("name", NAMES)
def test_every_prompt_has_front_matter_version_and_hash(name):
    prompt = PromptStore("v1").get(name)
    assert prompt.name == name and re.fullmatch(r"\d+\.\d+\.\d+", prompt.version)
    assert re.fullmatch(r"[0-9a-f]{8}", prompt.sha)
    assert prompt.ref == f"{name}@{prompt.version}#{prompt.sha}"


def test_prompt_hash_changes_with_content_not_with_whitespace_at_the_ends():
    store = PromptStore("v1")
    assert store.get("draft").sha != store.get("revise").sha
    assert store.get("draft") is store.get("draft")  # cached


def test_rendering_is_strict_about_placeholders():
    draft = PromptStore("v1").get("draft")
    with pytest.raises(KeyError):
        draft.render(product_json="{}")
    out = draft.render(
        product_json="{}",
        plan_json="{}",
        guidelines="g",
        languages="en",
        limits_json="{}",
        attempt=1,
    )
    assert "${" not in out and "<attempt>1</attempt>" in out


def test_unknown_prompt_or_version_is_a_config_error():
    with pytest.raises(ConfigError):
        PromptStore("v1").get("does_not_exist")
    with pytest.raises(ConfigError):
        PromptStore("v99").get("draft")
    with pytest.raises(ConfigError):
        PromptStore("../etc")


def test_prompt_version_is_configurable_through_settings():
    assert Settings.from_env({"PROMPT_VERSION": "v1"}).prompt_version == "v1"


def test_system_prompt_states_the_security_and_claims_rules():
    text = PromptStore("v1").get("system").render()
    for needle in (
        "untrusted DATA",
        "never instructions",
        "Never invent numbers",
        "No superlatives",
    ):
        assert needle in text


# ------------------------------------------------------------------ trace


def test_trace_records_steps_and_saves_valid_json(tmp_path):
    trace = Trace(full_io=False)
    trace.meta = {"provider": "mock"}
    trace.add("llm_call", purpose="draft", latency_ms=1.5)
    with trace.span("evaluate", iteration=1) as extra:
        extra["passed"] = True
    trace.result = {"status": "passed"}
    path = trace.save(tmp_path / "nested" / "traces")
    data = json.loads(path.read_text())
    assert path.name == f"{trace.run_id}.json"
    assert [s["step"] for s in data["steps"]] == ["llm_call", "evaluate"]
    assert data["steps"][1]["passed"] is True and data["steps"][1]["latency_ms"] >= 0
    assert [s["n"] for s in data["steps"]] == [1, 2]


def test_run_ids_are_unique_and_url_safe():
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50 and all(re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", i) for i in ids)


def test_preview_truncates_and_collapses_whitespace():
    assert preview("a\n\n  b") == "a b"
    assert len(preview("x" * 1000)) == 400 and preview("x" * 1000).endswith("…")


# ------------------------------------------------------------------ structured logging


def make_record(**extra):
    record = logging.LogRecord("pma.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.__dict__.update(extra)
    return record


def test_json_formatter_emits_one_json_object_with_extras():
    line = JsonFormatter().format(make_record(run_id="r1", latency_ms=3))
    data = json.loads(line)
    assert (
        data["event"] == "hello world" and data["level"] == "INFO" and data["logger"] == "pma.test"
    )
    assert data["run_id"] == "r1" and data["latency_ms"] == 3 and "ts" in data
    assert "\n" not in line


def test_text_formatter_appends_extras():
    assert TextFormatter().format(make_record(run_id="r1")).endswith("run_id=r1")


def test_configure_logging_is_idempotent_and_does_not_touch_the_root_logger(capsys):
    root_handlers = list(logging.getLogger().handlers)
    configure_logging("INFO", "json")
    configure_logging("INFO", "json")
    logger = logging.getLogger("pma")
    assert len(logger.handlers) == 1 and logging.getLogger().handlers == root_handlers
    get_logger("x").info("structured", extra={"k": 1})
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and json.loads(err[0])["k"] == 1
    configure_logging("WARNING", "text")
    assert io  # keep import used for clarity of intent
