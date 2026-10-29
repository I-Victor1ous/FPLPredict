import json
import os
from pathlib import Path

import pytest

from fplpredict.pipeline.status import (
    PipelineLocked,
    build_pipeline_status,
    pipeline_lock,
    write_pipeline_status,
)


def test_build_pipeline_status_marks_failures():
    status = build_pipeline_status(
        ["Premier League"],
        {"Premier League": {"newest_completed_date": "2026-09-20"}},
        {"Premier League": "RuntimeError: boom"},
    )
    assert status["ok"] is False
    assert status["leagues"][0]["status"] == "failed"
    assert status["leagues"][0]["error"] == "RuntimeError: boom"


def test_pipeline_lock_blocks_second_holder(tmp_path):
    lock = tmp_path / "weekly.lock"
    with pipeline_lock(lock):
        with pytest.raises(PipelineLocked):
            with pipeline_lock(lock):
                pass


def test_stale_lock_is_reclaimed(tmp_path):
    lock = tmp_path / "weekly.lock"
    lock.write_text("999999999\n")
    old = os.stat(lock).st_mtime
    os.utime(lock, (old - 13 * 3600, old - 13 * 3600))
    with pipeline_lock(lock):
        pass
    assert not lock.exists()


def test_write_pipeline_status_roundtrip(tmp_path):
    path = tmp_path / "pipeline_status.json"
    payload = build_pipeline_status(["La Liga"], {}, {})
    write_pipeline_status(payload, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["ok"] is True
    assert loaded["leagues"][0]["slug"] == "la-liga"
