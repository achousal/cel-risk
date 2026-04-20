"""Draft gate `decision.md` artifacts (post-run reasoning) from ledger + observation."""

from __future__ import annotations

from pathlib import Path

from .models import Rulebook


def draft_decision(ledger_path: Path, observation_path: Path, rulebook: Rulebook) -> str:
    """Return a `decision.md` string pre-populated from ledger predictions and observation metrics.

    The draft contains frontmatter (gate, observed_at) and four required body
    sections (Predictions that held, Predictions that failed, Actual claim type,
    Locks passed forward, Predictions for v_{k+1}). Only factual content is
    filled in: each prediction is listed with its observed delta and CI, and
    an LLM-fill marker for reasoning. The advisor does NOT classify the actual
    claim type — that is an LLM reasoning step authored on top of this draft.
    """
    raise NotImplementedError("scaffold: implement in v0.2")
