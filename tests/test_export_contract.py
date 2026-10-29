"""Committed site JSON must match what the Next.js pages read (stdlib only; runs in CI data-check)."""

import json
from pathlib import Path

import pytest

from tests.contract import league_payload_problems, pipeline_status_problems

DATA_DIR = Path(__file__).resolve().parent.parent / "web" / "public" / "data"
MANIFEST = DATA_DIR / "manifest.json"


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("no exported site data")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_lists_existing_league_files():
    manifest = _manifest()
    assert manifest["leagues"], "manifest has no leagues"
    for league in manifest["leagues"]:
        assert (DATA_DIR / f"{league['slug']}.json").exists(), league["slug"]


@pytest.mark.parametrize("slug", [p.stem for p in sorted(DATA_DIR.glob("*.json"))])
def test_league_json_matches_site_contract(slug):
    if slug in {"manifest", "pipeline_status"}:
        pytest.skip("not a league file")
    payload = json.loads((DATA_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    assert league_payload_problems(payload) == []


def test_pipeline_status_shape():
    path = DATA_DIR / "pipeline_status.json"
    if not path.exists():
        pytest.skip("pipeline has not written a status file yet")
    status = json.loads(path.read_text(encoding="utf-8"))
    assert pipeline_status_problems(status) == []
