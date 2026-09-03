"""Alert generation.

Produces the two ranked outputs described in the build spec (the third,
the research dashboard/export, lives in alerts/dashboard.py):

  1. Immediate alert -- gated: fires only when surprise, confirmation,
     AND confidence all clear their thresholds, and there is no open
     verification flag.
  2. Morning brief -- ungated: every scored event, ranked by a composite
     of impact/surprise/confidence/confirmation, filtered to the
     watchlist (which is already true by construction -- ingestion only
     runs against watchlist tickers).

All generated text (headline, assessment, risk note) is templated from
scores, not free-generated -- auditable and reproducible from the stored
numbers, never an LLM inventing a narrative on top of `facts`.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from market_intel.models.event import Event
from market_intel.scoring.market_confirmation import persistence_ratio


@dataclass
class AlertThresholds:
    min_surprise: float = 0.25
    min_confidence: float = 0.5
    min_confirmation: float = 0.25


def composite_rank_score(event: Event) -> float:
    interp = event.interpretation
    parts = []
    if interp.surprise_score is not None:
        parts.append(abs(interp.surprise_score))
    if interp.confidence is not None:
        parts.append(interp.confidence)
    if interp.confirmation_score is not None:
        parts.append(interp.confirmation_score)
    if interp.novelty_score is not None:
        parts.append(interp.novelty_score)
    if not parts:
        return 0.0
    return round(sum(parts) / len(parts), 4)


def passes_immediate_alert_gate(event: Event, thresholds: AlertThresholds = AlertThresholds()) -> bool:
    interp = event.interpretation
    if interp.verification_flag:
        return False
    surprise_ok = interp.surprise_score is not None and abs(interp.surprise_score) >= thresholds.min_surprise
    confidence_ok = interp.confidence is not None and interp.confidence >= thresholds.min_confidence
    confirmation_ok = interp.confirmation_score is not None and interp.confirmation_score >= thresholds.min_confirmation
    return surprise_ok and confidence_ok and confirmation_ok


def build_expectation_gap_text(event: Event) -> str:
    f = event.facts
    parts = []
    if f.eps_actual is not None and f.eps_consensus is not None:
        parts.append(f"EPS {f.eps_actual:.2f} vs. consensus {f.eps_consensus:.2f}")
    if f.revenue_actual is not None and f.revenue_consensus is not None:
        parts.append(f"Revenue {f.revenue_actual:,.0f} vs. consensus {f.revenue_consensus:,.0f}")
    if f.guidance_direction:
        parts.append(f"Guidance {f.guidance_direction}")
    return "; ".join(parts) if parts else "No structured expectation data extracted for this event."


def build_reaction_summary(event: Event) -> str:
    mr = event.market_reaction
    bits = []
    for label, val in (("5m", mr.return_5m), ("1h", mr.return_1h), ("1d", mr.return_1d), ("5d", mr.return_5d)):
        if val is not None:
            bits.append(f"{label} {val:+.2%}")
    if mr.volume_zscore is not None:
        bits.append(f"volume z={mr.volume_zscore:.1f}")
    if mr.iv_change is not None:
        bits.append(f"IV {mr.iv_change:+.2%}")
    return ", ".join(bits) if bits else "No market reaction data captured yet."


def build_headline(event: Event) -> str:
    label = event.event_type.value.replace("_", " ").title()
    direction = ""
    if event.interpretation.forward_earnings_effect:
        direction = f" ({event.interpretation.forward_earnings_effect})"
    return f"{event.company} ({event.ticker}): {label}{direction}"


def build_assessment(event: Event) -> str:
    interp = event.interpretation
    surprise = interp.surprise_score or 0.0
    confirmation = interp.confirmation_score or 0.0
    direction = "positive" if surprise > 0 else "negative" if surprise < 0 else "mixed"

    if confirmation >= 0.6:
        confirm_txt = "with a strong, confirmed market reaction"
    elif confirmation >= 0.3:
        confirm_txt = "with a moderate market reaction so far"
    else:
        confirm_txt = "with little market confirmation so far"

    return f"{direction.capitalize()} surprise {confirm_txt}."


def build_risk_note(event: Event) -> str:
    interp = event.interpretation
    if event.source_tier.value != "primary":
        return "Not yet confirmed by a primary source (SEC filing / IR release); treat as provisional."
    if interp.confidence is not None and interp.confidence < 0.5:
        return "Low confidence: thin corroboration or uncertain extraction/timestamp -- verify before acting."
    if interp.confirmation_score is not None and interp.confirmation_score < 0.25:
        return "Reaction may fade without further confirmation; watch for reversal."
    if len(event.evidence) <= 1:
        return "Single-source event; corroboration pending."
    return "Standard event risk; monitor for revision or reversal."


def priced_in_signal(event: Event) -> bool | None:
    """True if the initial (5m) move already captured most of the day's
    move -- i.e. little incremental drift, so acting on the headline
    now is less likely to add value."""
    ratio = persistence_ratio(event.market_reaction.return_5m, event.market_reaction.return_1d)
    if ratio is None:
        return None
    return 0.85 <= ratio <= 1.15


def build_alert(event: Event, alert_type: str, published_at: str | None = None) -> dict:
    interp = event.interpretation
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    latency_seconds = None
    if published_at:
        try:
            from market_intel.processing.timestamps import parse_utc

            latency_seconds = (parse_utc(generated_at) - parse_utc(published_at)).total_seconds()
        except ValueError:
            latency_seconds = None

    return {
        "alert_id": str(uuid.uuid4()),
        "event_id": event.event_id,
        "alert_type": alert_type,
        "generated_at": generated_at,
        "published_at": published_at,
        "surprise_score": interp.surprise_score,
        "confidence_score": interp.confidence,
        "confirmation_score": interp.confirmation_score,
        "composite_rank_score": composite_rank_score(event),
        "headline": build_headline(event),
        "expectation_gap": build_expectation_gap_text(event),
        "reaction_summary": build_reaction_summary(event),
        "assessment": build_assessment(event),
        "risk_note": build_risk_note(event),
        "sources": [e.url for e in event.evidence if e.url],
        "_latency_seconds": latency_seconds,  # convenience for evaluation logging; not a DB column
    }


def build_immediate_alerts(events: list[Event], thresholds: AlertThresholds = AlertThresholds()) -> list[dict]:
    alerts = [
        build_alert(e, "immediate", published_at=e.evidence[0].published_at if e.evidence else None)
        for e in events
        if passes_immediate_alert_gate(e, thresholds)
    ]
    alerts.sort(key=lambda a: a["composite_rank_score"], reverse=True)
    return alerts


def build_morning_brief(events: list[Event], top_n: int = 10) -> list[dict]:
    alerts = [build_alert(e, "morning_brief") for e in events]
    for alert, event in zip(alerts, events):
        alert["already_priced_in"] = priced_in_signal(event)
    alerts.sort(key=lambda a: a["composite_rank_score"], reverse=True)
    return alerts[:top_n]


def persist_alert(conn, alert: dict) -> None:
    from market_intel.db.database import dumps

    conn.execute(
        """
        INSERT INTO alerts
            (alert_id, event_id, alert_type, generated_at, published_at,
             surprise_score, confidence_score, confirmation_score, composite_rank_score,
             headline, expectation_gap, reaction_summary, assessment, risk_note, sources)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert["alert_id"], alert["event_id"], alert["alert_type"], alert["generated_at"], alert["published_at"],
            alert["surprise_score"], alert["confidence_score"], alert["confirmation_score"], alert["composite_rank_score"],
            alert["headline"], alert["expectation_gap"], alert["reaction_summary"], alert["assessment"], alert["risk_note"],
            dumps(alert["sources"]),
        ),
    )
