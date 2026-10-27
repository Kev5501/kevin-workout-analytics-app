"""Who may use the app and how many questions they get. No Streamlit imports, so it is easy to test.

Two roles, two passwords:
  * Guest (GUEST_PASSWORD): limited to GUEST_QUESTION_LIMIT questions per session, and to GUEST_DAILY_LIMIT
    questions per day across all guests (a page refresh starts a new session, so the daily cap is the safety net).
  * Kevin (APP_PASSWORD): no limit.
"""
from __future__ import annotations

import hmac
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

GUEST = "guest"
KEVIN = "kevin"
ROLE_CHOICES = {"A guest": GUEST, "Kevin": KEVIN}

LIMIT_MESSAGE = "Sorry, question limit reached, I don't want to run out of free tokens lol"


def verify_password(role: str, entered: str, *, kevin_password: str | None, guest_password: str | None) -> bool:
    """Constant-time check of ``entered`` against the password for ``role``. An unset/empty password never matches."""
    expected = {KEVIN: kevin_password, GUEST: guest_password}.get(role)
    if not expected or not entered:
        return False
    return hmac.compare_digest(entered.encode("utf-8"), str(expected).encode("utf-8"))


class DailyCounter:
    """Thread-safe count of guest questions today (UTC). Resets itself when the date changes."""

    def __init__(self, today=None):
        self._today = today or (lambda: datetime.now(timezone.utc).date())
        self._lock = threading.Lock()
        self._day = self._today()
        self._used = 0

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used

    def try_take(self, limit: int) -> bool:
        """Count one question if the daily limit allows it. A limit of 0 disables the cap (still counts)."""
        with self._lock:
            self._roll()
            if limit and self._used >= limit:
                return False
            self._used += 1
            return True


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    used: int  # questions this session has used after the check


def check_question(role: str, used_in_session: int, daily: DailyCounter, *, session_limit: int, daily_limit: int) -> Verdict:
    """Decide whether a question may run. Denied questions are not counted."""
    if role == KEVIN:
        return Verdict(True, used_in_session)
    if used_in_session >= session_limit:
        return Verdict(False, used_in_session)
    if not daily.try_take(daily_limit):
        return Verdict(False, used_in_session)
    return Verdict(True, used_in_session + 1)
