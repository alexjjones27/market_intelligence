"""Novelty scoring: is this the first disclosure of an issue, a material
change from prior management language, an unexpected guidance
withdrawal, etc.

MVP approach (per spec): a simple diff against the company's prior N
events of the same type. Two signals, blended:
  1. Guidance-direction shift (raised -> lowered/withdrawn is maximally
     novel; same direction as last time is not novel).
  2. Text similarity of this event's evidence titles against the prior
     events' -- low similarity to anything recent = more novel. Uses
     `difflib` (same simple approach as dedup's near-duplicate check;
     an embedding-based version is a natural upgrade, not needed for
     MVP scale).
"""
from __future__ import annotations

import difflib

GUIDANCE_SHIFT_NOVELTY = {
    # (previous, current) -> novelty in [0, 1]
    ("raised", "raised"): 0.1,
    ("raised", "maintained"): 0.4,
    ("raised", "lowered"): 0.9,
    ("raised", "withdrawn"): 1.0,
    ("maintained", "raised"): 0.5,
    ("maintained", "maintained"): 0.0,
    ("maintained", "lowered"): 0.7,
    ("maintained", "withdrawn"): 1.0,
    ("lowered", "raised"): 0.8,
    ("lowered", "maintained"): 0.4,
    ("lowered", "lowered"): 0.2,
    ("lowered", "withdrawn"): 0.9,
    ("withdrawn", "raised"): 0.6,
    ("withdrawn", "maintained"): 0.5,
    ("withdrawn", "lowered"): 0.5,
    ("withdrawn", "withdrawn"): 0.1,
}


def guidance_shift_novelty(current_direction: str | None, most_recent_prior_direction: str | None) -> float | None:
    if current_direction is None:
        return None
    if most_recent_prior_direction is None:
        return 0.5  # no prior guidance on record -> can't assess a shift, stay neutral
    return GUIDANCE_SHIFT_NOVELTY.get((most_recent_prior_direction, current_direction), 0.5)


def text_novelty(current_text: str | None, prior_texts: list[str]) -> float | None:
    """1 - max similarity vs. the most similar prior text. None if we
    have nothing to compare against."""
    clean_priors = [t for t in prior_texts if t]
    if not current_text or not clean_priors:
        return None
    max_sim = max(
        difflib.SequenceMatcher(None, current_text.lower(), t.lower()).ratio() for t in clean_priors
    )
    return round(1 - max_sim, 4)


def compute_novelty_score(
    is_first_disclosure_of_type: bool,
    guidance_direction: str | None = None,
    most_recent_prior_guidance_direction: str | None = None,
    current_text: str | None = None,
    prior_texts: list[str] | None = None,
) -> float:
    if is_first_disclosure_of_type:
        return 1.0

    components = []
    g = guidance_shift_novelty(guidance_direction, most_recent_prior_guidance_direction)
    if g is not None:
        components.append(g)
    t = text_novelty(current_text, prior_texts or [])
    if t is not None:
        components.append(t)

    if not components:
        return 0.5  # nothing to compare against -> neutral, not "not novel"
    return round(sum(components) / len(components), 4)
