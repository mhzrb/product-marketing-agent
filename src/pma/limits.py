"""Channel length limits (characters), mirrored in rag/guidelines/channel_formats.md."""

from __future__ import annotations

# kind -> (min_chars, max_chars)
LIMITS: dict[str, tuple[int, int]] = {
    "description": (60, 600),
    "ad_copy": (20, 150),
    "social_post": (30, 280),
    "email_subject": (10, 60),
}
