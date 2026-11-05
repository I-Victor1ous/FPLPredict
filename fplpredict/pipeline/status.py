"""Per-league health summary written alongside the site JSON."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fplpredict.paths import LOG_DIR, WEB_DATA_DIR, league_slug

PIPELINE_STATUS_JSON = WEB_DATA_DIR / "pipeline_status.json"
WEEKLY_LOCK = LOG_DIR / "weekly.lock"
PARTIAL_FAILURE_EXIT = 2
STALE_LOCK_SECONDS = 12 * 3600


def build_pipeline_status(
    leagues: list[str],
    reports: dict[str, dict],
    errors: dict[str, str],
    *,
    run_at: datetime | None = None,
) -> dict:
    """``reports`` maps league -> data-quality stats; ``errors`` maps league -> failure message."""
    run_at = run_at or datetime.now(timezone.utc)
    entries = []
    for league in leagues:
        stats = reports.get(league, {})
        error = errors.get(league)
        entries.append(
            {
                "name": league,
                "slug": league_slug(league),
                "status": "failed" if error else "ok",
                "error": error,
                "newestCompletedDate": stats.get("newest_completed_date"),
                "xgMissingRate": stats.get("latest_season_xG_missing_rate"),
                "upcomingFixtures": stats.get("future_rows"),
            }
        )
    return {
        "runAt": run_at.isoformat(),
        "ok": not errors,
        "leagues": entries,
    }


def write_pipeline_status(status: dict, path: Path = PIPELINE_STATUS_JSON) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return path


class PipelineLocked(RuntimeError):
    pass


@contextmanager
def pipeline_lock(path: Path = WEEKLY_LOCK):
    """Exclusive lock file; a stale lock from a dead process is reclaimed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = _lock_holder(path)
        if holder is not None and _lock_is_live(path, holder):
            raise PipelineLocked(f"Weekly pipeline already running (pid {holder}).") from None
        path.unlink(missing_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def _lock_holder(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _lock_is_live(path: Path, pid: int) -> bool:
    # os.kill(pid, 0) sends CTRL_C_EVENT on Windows, so fall back to lock age there.
    if os.name == "nt":
        return time.time() - path.stat().st_mtime < STALE_LOCK_SECONDS
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
