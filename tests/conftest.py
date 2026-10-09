import pytest

from workout_chat import config
from workout_chat.data import EXPECTED_COLUMNS, clean_dataframe

HEADER = list(EXPECTED_COLUMNS)


@pytest.fixture(autouse=True)
def isolate_from_real_secrets(monkeypatch):
    """Tests must never read the developer's real .streamlit/secrets.toml."""
    monkeypatch.setattr(config, "_streamlit_secret", lambda name: None)


def make_row(date="2025-03-14", set_id="S1", day="Push", ex="Bench Press", typ="Hypertrophy",
             n=1, w=135, reps=8, rpe=8.5, norm=170.2, bw=180.4, comment="felt good"):
    return [date, set_id, day, ex, typ, n, w, reps, rpe, norm, bw, comment]


@pytest.fixture
def sample_rows_raw():
    """Raw sheet rows: oldest set is 'OLDEST'; Bench Press is the most frequent exercise."""
    return [
        HEADER,
        make_row("2025-03-05", "OLDEST", comment="SECRET shoulder tweak on last rep"),
        make_row("2025-03-05", "S2", comment="SECRET shoulder tweak on last rep"),
        make_row("2025-03-12", "S3", day="Legs", ex="Squat", typ="Strength", w=225, reps=5, rpe="", comment="great pump"),
        make_row("2025-03-12", "S4", day="Legs", ex="Squat", typ="Strength", w=225, reps=5, rpe=9, comment="great pump"),
        make_row("2025-03-14", "S5", comment=""),
        make_row("2025-03-14", "S6", comment=""),
    ]


@pytest.fixture
def sample_df(sample_rows_raw):
    return clean_dataframe(sample_rows_raw)