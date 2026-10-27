from datetime import date

import pytest

from workout_chat import access
from workout_chat.access import DailyCounter, check_question, verify_password

PW = {"kevin_password": "kev-pw", "guest_password": "guest-pw"}


@pytest.mark.parametrize("role,entered,ok", [
    (access.KEVIN, "kev-pw", True),
    (access.GUEST, "guest-pw", True),
    (access.KEVIN, "guest-pw", False),      # the passwords are not interchangeable
    (access.GUEST, "kev-pw", False),
    (access.KEVIN, "", False),
    (access.KEVIN, "KEV-PW", False),
    (access.KEVIN, "kev-pw ", False),
    ("admin", "kev-pw", False),
])
def test_verify_password(role, entered, ok):
    assert verify_password(role, entered, **PW) is ok


def test_unset_or_empty_password_never_matches():
    assert not verify_password(access.KEVIN, "", kevin_password="", guest_password="x")
    assert not verify_password(access.GUEST, "anything", kevin_password="x", guest_password=None)


def test_non_ascii_password_does_not_crash():
    assert verify_password(access.KEVIN, "pässwörd", kevin_password="pässwörd", guest_password="x")
    assert not verify_password(access.KEVIN, "pässwörd", kevin_password="password", guest_password="x")


def test_role_choices_match_the_prompt():
    assert list(access.ROLE_CHOICES) == ["A guest", "Kevin"]


def test_limit_message_is_exact():
    assert access.LIMIT_MESSAGE == "Sorry, question limit reached, I don't want to run out of free tokens lol"


# ---- the per-session limit ---------------------------------------------------------------
def test_guest_gets_exactly_five_questions():
    daily = DailyCounter()
    used = 0
    for _ in range(5):
        verdict = check_question(access.GUEST, used, daily, session_limit=5, daily_limit=0)
        assert verdict.allowed
        used = verdict.used
    assert used == 5
    denied = check_question(access.GUEST, used, daily, session_limit=5, daily_limit=0)
    assert not denied.allowed and denied.used == 5


def test_kevin_has_no_limit_and_does_not_use_the_daily_budget():
    daily = DailyCounter()
    for i in range(100):
        assert check_question(access.KEVIN, i, daily, session_limit=5, daily_limit=5).allowed
    assert daily.used == 0


# ---- the daily limit -----------------------------------------------------------------------
def test_daily_limit_is_shared_between_guest_sessions():
    daily = DailyCounter()
    assert all(check_question(access.GUEST, 0, daily, session_limit=5, daily_limit=3).allowed for _ in range(3))
    assert not check_question(access.GUEST, 0, daily, session_limit=5, daily_limit=3).allowed


def test_denied_questions_are_not_counted():
    daily = DailyCounter()
    check_question(access.GUEST, 5, daily, session_limit=5, daily_limit=10)
    assert daily.used == 0


def test_daily_limit_zero_disables_the_cap():
    daily = DailyCounter()
    assert all(check_question(access.GUEST, 0, daily, session_limit=5, daily_limit=0).allowed for _ in range(50))


def test_daily_counter_resets_on_a_new_day():
    day = [date(2026, 10, 20)]
    daily = DailyCounter(today=lambda: day[0])
    assert daily.try_take(1) and not daily.try_take(1)
    day[0] = date(2026, 10, 21)
    assert daily.used == 0 and daily.try_take(1)
