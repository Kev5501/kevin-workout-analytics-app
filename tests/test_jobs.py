"""Background turns: progress labels, saving the reply, failures, one turn per conversation."""
import threading

import pytest

from workout_chat.graph import GENERIC_ERROR
from workout_chat.jobs import DONE, FAILED, RUNNING, JobRunner


class SyncExecutor:
    """Runs submitted work immediately (keeps tests deterministic)."""

    def submit(self, fn, *args):
        fn(*args)


class HeldExecutor:
    """Holds submitted work until release() (simulates a turn still running)."""

    def __init__(self):
        self.held = []

    def submit(self, fn, *args):
        self.held.append((fn, args))

    def release(self):
        while self.held:
            fn, args = self.held.pop(0)
            fn(*args)


def test_reply_is_saved_and_the_job_finishes():
    saved = []
    runner = JobRunner(SyncExecutor())

    def work(on_step):
        on_step("Writing and running code...")
        return {"role": "assistant", "content": "6 sets"}

    job = runner.submit("t1", "How many sets?", work, saved.append)
    assert saved == [{"role": "assistant", "content": "6 sets"}]
    assert job.status == DONE and job.label == "Writing and running code..." and job.message["content"] == "6 sets"
    assert runner.get("t1") is job and not runner.is_running("t1")


def test_a_running_job_is_visible_and_blocks_a_second_question():
    ex = HeldExecutor()
    runner = JobRunner(ex)
    runner.submit("t1", "q", lambda step: {"role": "assistant", "content": "a"}, lambda m: None)
    assert runner.is_running("t1") and runner.get("t1").status == RUNNING
    with pytest.raises(RuntimeError):
        runner.submit("t1", "again", lambda step: {}, lambda m: None)
    runner.submit("t2", "other conversation", lambda step: {}, lambda m: None)   # other threads are fine
    ex.release()
    assert not runner.is_running("t1")
    runner.submit("t1", "next", lambda step: {"role": "assistant", "content": "b"}, lambda m: None)


def test_crashing_work_still_gives_a_reply():
    saved = []

    def boom(step):
        raise ValueError("bug")

    job = JobRunner(SyncExecutor()).submit("t1", "q", boom, saved.append)
    assert job.status == FAILED and saved[0]["content"] == GENERIC_ERROR and "ValueError" in saved[0]["debug"]


def test_a_failed_save_keeps_the_reply_on_the_job():
    def bad_save(message):
        raise ConnectionError("firestore down")

    job = JobRunner(SyncExecutor()).submit("t1", "q", lambda s: {"role": "assistant", "content": "a"}, bad_save)
    assert job.status == DONE and job.message["content"] == "a" and "firestore down" in job.save_error


def test_status_flips_only_after_the_reply_is_saved():
    seen = {}
    runner = JobRunner(SyncExecutor())

    def save(message):
        seen["status_during_save"] = runner.get("t1").status

    runner.submit("t1", "q", lambda s: {"role": "assistant", "content": "a"}, save)
    assert seen["status_during_save"] == RUNNING


def test_finished_jobs_are_forgotten_after_a_while():
    now = [1000.0]
    runner = JobRunner(SyncExecutor(), keep_seconds=600, clock=lambda: now[0])
    runner.submit("t1", "q", lambda s: {"role": "assistant", "content": "a"}, lambda m: None)
    now[0] += 599
    assert runner.get("t1") is not None
    now[0] += 2
    assert runner.get("t1") is None


def test_real_threads_run_in_the_background():
    gate, done = threading.Event(), threading.Event()
    runner = JobRunner()

    def work(step):
        gate.wait(5)
        return {"role": "assistant", "content": "a"}

    runner.submit("t1", "q", work, lambda m: done.set())
    assert runner.is_running("t1")          # submit returned while the turn is still going
    gate.set()
    assert done.wait(5)
