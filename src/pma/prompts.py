"""Versioned prompt files.

Prompts live in ``src/pma/prompts/<version_dir>/<name>.md`` with a small front-matter block::

    ---
    name: draft
    version: 1.0.0
    ---
    template text using ${placeholders}

``Prompt.ref`` (``draft@1.0.0#a1b2c3d4``) is written into every trace step, so a trace always says
exactly which prompt text produced a given output (the hash changes if the file is edited without
bumping ``version``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from string import Template

from pma.errors import ConfigError

_FRONT = re.compile(r"\A---\n(.*?)\n---\n(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    template: str
    sha: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}#{self.sha}"

    def render(self, **values: object) -> str:
        """Strict substitution: a missing placeholder raises instead of leaking ``${x}``."""
        return Template(self.template).substitute({k: str(v) for k, v in values.items()})


class PromptStore:
    def __init__(self, version_dir: str = "v1") -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", version_dir):
            raise ConfigError(f"invalid PROMPT_VERSION {version_dir!r}")
        self.version_dir = version_dir
        self._cache: dict[str, Prompt] = {}

    def get(self, name: str) -> Prompt:
        if name in self._cache:
            return self._cache[name]
        path = resources.files("pma") / "prompts" / self.version_dir / f"{name}.md"
        try:
            raw = path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError) as exc:
            raise ConfigError(
                f"prompt {name!r} not found for version {self.version_dir!r}"
            ) from exc
        match = _FRONT.match(raw)
        if not match:
            raise ConfigError(f"prompt file {name}.md is missing its front-matter block")
        meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
        meta = {k.strip(): v.strip() for k, v in meta.items()}
        body = match.group(2).strip() + "\n"
        prompt = Prompt(
            name=meta.get("name", name),
            version=meta.get("version", "0"),
            template=body,
            sha=hashlib.sha256(body.encode()).hexdigest()[:8],
        )
        self._cache[name] = prompt
        return prompt
