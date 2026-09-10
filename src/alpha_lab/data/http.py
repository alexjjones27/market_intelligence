"""Shared HTTP helper with backoff.

Every upstream here is a free, rate-limited public endpoint (Wikipedia's API,
SEC's data.sec.gov, Yahoo's chart service). All three answer a burst with 429 or
403 rather than queueing, so a naive loop over 500 tickers fails most of them.
This centralises polite retry behaviour: exponential backoff, ``Retry-After``
when the server sends one, and a hard ceiling so a run cannot hang forever.
"""
from __future__ import annotations

import logging
import random
import time

import requests

logger = logging.getLogger(__name__)

RETRY_STATUS = {403, 408, 425, 429, 500, 502, 503, 504}


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 45,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(base_delay * (2**attempt), max_delay)
                )
                delay += random.uniform(0, 0.5)  # de-synchronise repeated bursts
                if attempt < max_attempts - 1:
                    logger.debug(
                        "%s -> %s, retrying in %.1fs (attempt %d/%d)",
                        url, resp.status_code, delay, attempt + 1, max_attempts,
                    )
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay) + random.uniform(0, 0.5)
            time.sleep(delay)
    raise RuntimeError(f"GET failed after {max_attempts} attempts: {url}") from last_exc
