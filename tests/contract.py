"""Shape checks for the site JSON (mirrors ``LeagueData`` in web/lib/types.ts)."""

from __future__ import annotations

import math

LEAGUE_KEYS = {"upcoming", "awaitingResult", "history", "weeklyPerformance", "modelTimeline", "meta"}
FIXTURE_KEYS = {"date", "team", "opponent", "pred", "pickLabel", "probs"}
PROB_KEYS = {"teamWin", "draw", "oppWin"}


def league_payload_problems(payload: dict) -> list[str]:
    problems = [f"missing key {k}" for k in sorted(LEAGUE_KEYS - set(payload))]
    for section in ("upcoming", "awaitingResult", "history"):
        items = payload.get(section)
        if not isinstance(items, list):
            problems.append(f"{section} is not a list")
            continue
        for i, item in enumerate(items):
            missing = FIXTURE_KEYS - set(item)
            if missing:
                problems.append(f"{section}[{i}] missing {sorted(missing)}")
                continue
            probs = item["probs"]
            if set(probs) != PROB_KEYS or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in probs.values()
            ):
                problems.append(f"{section}[{i}] has invalid probs {probs}")
            elif abs(sum(probs.values()) - 100) > 1.5:
                problems.append(f"{section}[{i}] probs sum to {sum(probs.values()):.2f}, not 100")
    for i, week in enumerate(payload.get("weeklyPerformance") or []):
        if not {"weekStart", "n", "correct"} <= set(week):
            problems.append(f"weeklyPerformance[{i}] missing fields")
    for i, seg in enumerate(payload.get("modelTimeline") or []):
        if not {"from", "strategy"} <= set(seg):
            problems.append(f"modelTimeline[{i}] missing fields")
    if not isinstance(payload.get("meta"), dict):
        problems.append("meta is not an object")
    return problems


def pipeline_status_problems(status: dict) -> list[str]:
    problems = [f"missing key {k}" for k in ("runAt", "ok", "leagues") if k not in status]
    for i, league in enumerate(status.get("leagues") or []):
        if league.get("status") not in {"ok", "failed"}:
            problems.append(f"leagues[{i}] has status {league.get('status')!r}")
        if not league.get("slug"):
            problems.append(f"leagues[{i}] missing slug")
    return problems
