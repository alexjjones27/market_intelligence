"""Event-type classification.

Two signals, reconciled in priority order:
  1. A structured guess already attached upstream (e.g. SEC 8-K item
     codes -- see ingestion/sec_edgar.py). Trusted first since it comes
     from a controlled vocabulary, not free text.
  2. Keyword classification over title + body text, used whenever (1) is
     absent, "other", or the item is a news article.

MVP simplification: single-label, first-match-wins keyword rules. Good
enough to route items into the six in-scope event types; a production
version would use a trained classifier or LLM extraction constrained to
the same label set.
"""
from __future__ import annotations

import re

from market_intel.models.event import EventType

# Ordered: more specific / higher-precision patterns first.
_KEYWORD_RULES: list[tuple[EventType, re.Pattern]] = [
    (
        EventType.M_AND_A,
        re.compile(r"\b(acqui(re|res|sition)|merger|to (buy|acquire)|divest(iture)?|takeover|tender offer)\b", re.I),
    ),
    (
        EventType.EXEC_CHANGE,
        re.compile(
            r"\b(chief executive|chief financial|\bceo\b|\bcfo\b|\bcoo\b|resigns?|steps down|"
            r"appoints? .*(ceo|cfo|coo|president)|names? new (ceo|cfo|coo)|departure of)\b",
            re.I,
        ),
    ),
    (
        EventType.LEGAL_REGULATORY,
        re.compile(r"\b(lawsuit|litigation|investigation|sec charges|settlement|fine[ds]?|regulatory|subpoena|recall)\b", re.I),
    ),
    (
        EventType.CAPITAL_ALLOCATION,
        re.compile(r"\b(share (repurchase|buyback)|buyback program|special dividend|dividend increase|raises? dividend)\b", re.I),
    ),
    (
        EventType.GUIDANCE_CHANGE,
        re.compile(r"\b(guidance|forecast|outlook)\b.*\b(raise[sd]?|lower[sd]?|cut|withdraw[sn]?|reaffirm)\b", re.I),
    ),
    (
        EventType.EARNINGS,
        re.compile(r"\b(earnings|quarterly results|eps|revenue|q[1-4] results|reports (first|second|third|fourth) quarter)\b", re.I),
    ),
]


def classify_from_text(title: str | None, body_text: str | None) -> EventType:
    text = f"{title or ''} {body_text or ''}"
    for event_type, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            return event_type
    return EventType.OTHER


def classify(event_type_guess: str | None, title: str | None, body_text: str | None) -> EventType:
    if event_type_guess:
        try:
            candidate = EventType(event_type_guess)
            if candidate is not EventType.OTHER:
                return candidate
        except ValueError:
            pass
    return classify_from_text(title, body_text)
