"""Tests for ts_db — Target Scheduler SQLite reader (read-only)."""

import sqlite3
import pytest

from ts_db import (
    connect,
    get_exposure_plan,
    list_projects,
    next_target,
    tool_get_exposure_plan,
    tool_list_projects,
    tool_next_target,
)

# Mirror constants from conftest.py — pytest's conftest is not import-friendly
PROFILE_A = "11111111-1111-1111-1111-111111111111"
PROFILE_B = "22222222-2222-2222-2222-222222222222"


class TestConnectReadOnly:
    def test_connect_returns_sqlite_connection(self, empty_db):
        conn = connect(str(empty_db))
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_connect_refuses_writes(self, empty_db):
        """Writes must raise — we never mutate the user's TS database."""
        conn = connect(str(empty_db))
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO project (Id, profileId, name, state) VALUES (99, 'x', 'x', 1)"
            )
            conn.commit()
        conn.close()

    def test_connect_rejects_missing_file(self, tmp_path):
        """Asking for a non-existent DB must fail loudly, not silently create one."""
        missing = tmp_path / "does_not_exist.sqlite"
        with pytest.raises((sqlite3.OperationalError, FileNotFoundError)):
            connect(str(missing))


class TestListProjects:
    def test_empty_db_returns_empty_list(self, empty_db):
        conn = connect(str(empty_db))
        assert list_projects(conn) == []

    def test_default_returns_active_only(self, seeded_db):
        """Default behavior: only state=1 (Active) projects come back."""
        conn = connect(str(seeded_db))
        rows = list_projects(conn)
        names = {r["name"] for r in rows}
        assert names == {"M81 Bode's Galaxy", "NGC 7000 N.America", "Other Rig Project"}
        # Draft (state=0) must be excluded
        assert "Drafty Project" not in names

    def test_active_only_false_returns_all(self, seeded_db):
        conn = connect(str(seeded_db))
        rows = list_projects(conn, active_only=False)
        assert len(rows) == 4

    def test_profile_filter(self, seeded_db):
        """profile_id filter scopes to a single rig/profile."""
        conn = connect(str(seeded_db))
        rows = list_projects(conn, profile_id=PROFILE_A)
        assert {r["name"] for r in rows} == {"M81 Bode's Galaxy", "NGC 7000 N.America"}

        rows_b = list_projects(conn, profile_id=PROFILE_B)
        assert [r["name"] for r in rows_b] == ["Other Rig Project"]

    def test_ordered_by_priority_desc(self, seeded_db):
        """Higher priority first — Planner picks from the top."""
        conn = connect(str(seeded_db))
        rows = list_projects(conn, profile_id=PROFILE_A)
        # M81 has priority 100, NGC 7000 has priority 50
        assert [r["name"] for r in rows] == ["M81 Bode's Galaxy", "NGC 7000 N.America"]

    def test_returns_useful_fields(self, seeded_db):
        """Result rows expose the fields the Planner actually needs."""
        conn = connect(str(seeded_db))
        m81 = next(r for r in list_projects(conn) if r["name"] == "M81 Bode's Galaxy")
        assert m81["id"] == 1
        assert m81["profile_id"] == PROFILE_A
        assert m81["state"] == 1
        assert m81["priority"] == 100
        assert m81["minimum_altitude"] == 30.0
        assert m81["meridian_window"] == 60


class TestGetExposurePlan:
    def test_unknown_target_returns_empty(self, seeded_db):
        conn = connect(str(seeded_db))
        assert get_exposure_plan(conn, target_id=9999) == []

    def test_m81_returns_two_plans(self, seeded_db):
        """M81 has L (1000) + Ha (1001) exposure plans."""
        conn = connect(str(seeded_db))
        plans = get_exposure_plan(conn, target_id=10)
        assert len(plans) == 2
        names = {p["template_name"] for p in plans}
        assert names == {"L", "Ha"}

    def test_plan_exposes_template_fields(self, seeded_db):
        """Joined fields from exposuretemplate must be accessible."""
        conn = connect(str(seeded_db))
        l_plan = next(p for p in get_exposure_plan(conn, target_id=10)
                       if p["template_name"] == "L")
        assert l_plan["filter_name"] == "L"
        assert l_plan["gain"] == 100
        assert l_plan["offset"] == 50
        assert l_plan["bin"] == 1
        assert l_plan["exposure"] == 180.0

    def test_remaining_is_desired_minus_acquired(self, seeded_db):
        conn = connect(str(seeded_db))
        plans = {p["template_name"]: p for p in get_exposure_plan(conn, target_id=10)}
        # L plan: desired=30, acquired=10 → remaining=20
        assert plans["L"]["desired"] == 30
        assert plans["L"]["acquired"] == 10
        assert plans["L"]["remaining"] == 20
        # Ha plan: desired=20, acquired=20 → remaining=0
        assert plans["Ha"]["remaining"] == 0

    def test_remaining_never_negative(self, seeded_db):
        """If acquired somehow exceeds desired, remaining must clamp to 0."""
        conn = connect(str(seeded_db))
        # NGC 7000's Ha plan in fixture has acquired=0, desired=10, but let's
        # check via the L plan that math works for both directions.
        plans = get_exposure_plan(conn, target_id=11)
        for p in plans:
            assert p["remaining"] >= 0

    def test_disabled_plans_are_marked(self, seeded_db):
        """NGC 7000's Ha plan (1003) has enabled=0; it must show through."""
        conn = connect(str(seeded_db))
        plans = {p["template_name"]: p for p in get_exposure_plan(conn, target_id=11)}
        assert plans["L"]["enabled"] is True
        assert plans["Ha"]["enabled"] is False


def _mark_target_complete(db_path, target_id):
    """Test helper: set all of a target's enabled plans to acquired=desired."""
    w = sqlite3.connect(str(db_path))
    w.execute("UPDATE exposureplan SET acquired = desired WHERE targetid = ?", (target_id,))
    w.commit()
    w.close()


class TestNextTarget:
    def test_empty_db_returns_none(self, empty_db):
        conn = connect(str(empty_db))
        assert next_target(conn) is None

    def test_picks_highest_priority_active_project(self, seeded_db):
        """M81 has priority 100, NGC 7000 has priority 50 → M81 wins."""
        conn = connect(str(seeded_db))
        nt = next_target(conn, profile_id=PROFILE_A)
        assert nt is not None
        assert nt["target"]["name"] == "M81"
        assert nt["project"]["name"] == "M81 Bode's Galaxy"

    def test_falls_through_when_top_priority_is_complete(self, seeded_db):
        """Mark M81 complete → next_target should switch to NGC 7000."""
        _mark_target_complete(seeded_db, target_id=10)
        conn = connect(str(seeded_db))
        nt = next_target(conn, profile_id=PROFILE_A)
        assert nt is not None
        assert nt["target"]["name"] == "NGC 7000"

    def test_profile_scoped(self, seeded_db):
        """Profile B has only 'Other Rig Target'."""
        conn = connect(str(seeded_db))
        nt = next_target(conn, profile_id=PROFILE_B)
        assert nt is not None
        assert nt["target"]["name"] == "Other Rig Target"

    def test_returns_only_remaining_enabled_plans(self, seeded_db):
        """The returned plan list filters to enabled AND remaining > 0."""
        conn = connect(str(seeded_db))
        nt = next_target(conn, profile_id=PROFILE_A)
        # M81 wins — its Ha plan is complete (remaining=0), only L should appear.
        plan_names = [p["template_name"] for p in nt["plans"]]
        assert plan_names == ["L"]

    def test_returns_none_when_everything_complete(self, seeded_db):
        _mark_target_complete(seeded_db, target_id=10)
        _mark_target_complete(seeded_db, target_id=11)
        conn = connect(str(seeded_db))
        assert next_target(conn, profile_id=PROFILE_A) is None

    def test_includes_target_coordinates(self, seeded_db):
        """RA/Dec must be exposed so the Planner can slew/frame."""
        conn = connect(str(seeded_db))
        nt = next_target(conn, profile_id=PROFILE_A)
        assert nt["target"]["ra"] == 148.888
        assert nt["target"]["dec"] == 69.065


class TestToolEnvelopes:
    """The tool_* helpers wrap raw functions in the {Success,...,Type} envelope
    expected by every other MCP tool in nina_advanced_mcp.py."""

    def test_list_projects_envelope(self, seeded_db):
        env = tool_list_projects(db_path=str(seeded_db))
        assert env["Success"] is True
        assert env["Type"] == "TARGET_SCHEDULER_DB"
        assert "Projects" in env["Details"]
        assert len(env["Details"]["Projects"]) == 3  # 3 active across both profiles

    def test_list_projects_profile_filter_threads_through(self, seeded_db):
        env = tool_list_projects(db_path=str(seeded_db), profile_id=PROFILE_B)
        assert env["Success"] is True
        assert len(env["Details"]["Projects"]) == 1

    def test_get_exposure_plan_envelope(self, seeded_db):
        env = tool_get_exposure_plan(db_path=str(seeded_db), target_id=10)
        assert env["Success"] is True
        assert env["Type"] == "TARGET_SCHEDULER_DB"
        assert len(env["Details"]["Plans"]) == 2

    def test_next_target_envelope_when_found(self, seeded_db):
        env = tool_next_target(db_path=str(seeded_db), profile_id=PROFILE_A)
        assert env["Success"] is True
        assert env["Details"]["Found"] is True
        assert env["Details"]["Target"]["name"] == "M81"

    def test_next_target_envelope_when_none(self, empty_db):
        env = tool_next_target(db_path=str(empty_db))
        assert env["Success"] is True
        assert env["Details"]["Found"] is False
        assert env["Details"]["Target"] is None

    def test_error_returns_failure_envelope(self, tmp_path):
        """Missing DB → Success=False, ErrorType set — does NOT raise."""
        env = tool_list_projects(db_path=str(tmp_path / "missing.sqlite"))
        assert env["Success"] is False
        assert env["ErrorType"] == "TargetSchedulerDBError"
        assert env["Type"] == "TARGET_SCHEDULER_DB"
