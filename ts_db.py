"""Read-only access to the NINA Target Scheduler v5 SQLite database.

The orchestrator's Planner agent reads this DB to choose the next target.
Writes are *not* supported by design — NINA's own Target Scheduler integration
owns frame-acquired bookkeeping. We open with SQLite's read-only URI mode so
any accidental write raises OperationalError.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional


STATE_DRAFT = 0
STATE_ACTIVE = 1
STATE_INACTIVE = 2
STATE_CLOSED = 3


def _project_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["Id"],
        "profile_id": row["profileId"],
        "name": row["name"],
        "description": row["description"],
        "state": row["state"],
        "priority": row["priority"],
        "minimum_altitude": row["minimumaltitude"],
        "maximum_altitude": row["maximumAltitude"],
        "meridian_window": row["meridianwindow"],
    }


def default_db_path() -> Path:
    """Standard install location on Windows."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        raise RuntimeError("LOCALAPPDATA is not set — cannot locate Target Scheduler DB")
    return Path(base) / "NINA" / "SchedulerPlugin" / "schedulerdb.sqlite"


def connect(path: Optional[str | os.PathLike] = None) -> sqlite3.Connection:
    """Open the Target Scheduler DB read-only.

    Raises sqlite3.OperationalError if the file is missing.
    """
    db_path = Path(path) if path is not None else default_db_path()
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def list_projects(
    conn: sqlite3.Connection,
    *,
    profile_id: Optional[str] = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:
    """Return projects ordered by priority descending.

    By default only Active (state=1) projects are returned. Pass
    active_only=False to include Draft/Inactive/Closed projects too.
    """
    sql = "SELECT * FROM project WHERE 1=1"
    params: list[Any] = []
    if active_only:
        sql += " AND state = ?"
        params.append(STATE_ACTIVE)
    if profile_id is not None:
        sql += " AND profileId = ?"
        params.append(profile_id)
    sql += " ORDER BY priority DESC, Id ASC"
    return [_project_row_to_dict(r) for r in conn.execute(sql, params)]


def get_exposure_plan(conn: sqlite3.Connection, *, target_id: int) -> list[dict[str, Any]]:
    """Return the per-filter exposure plans for a target.

    Joins exposureplan with exposuretemplate so the Planner sees filter name,
    gain, offset, binning alongside desired/acquired counts.
    """
    sql = """
        SELECT
            ep.Id              AS plan_id,
            ep.exposure        AS exposure,
            ep.desired         AS desired,
            ep.acquired        AS acquired,
            ep.accepted        AS accepted,
            ep.enabled         AS enabled,
            et.name            AS template_name,
            et.filtername      AS filter_name,
            et.gain            AS gain,
            et.offset          AS offset,
            et.bin             AS bin
        FROM exposureplan ep
        LEFT JOIN exposuretemplate et ON et.Id = ep.exposureTemplateId
        WHERE ep.targetid = ?
        ORDER BY ep.Id ASC
    """
    rows = conn.execute(sql, (target_id,)).fetchall()
    plans = []
    for r in rows:
        desired = r["desired"] or 0
        acquired = r["acquired"] or 0
        plans.append({
            "plan_id": r["plan_id"],
            "template_name": r["template_name"],
            "filter_name": r["filter_name"],
            "exposure": r["exposure"],
            "gain": r["gain"],
            "offset": r["offset"],
            "bin": r["bin"],
            "desired": desired,
            "acquired": acquired,
            "accepted": r["accepted"] or 0,
            "remaining": max(desired - acquired, 0),
            "enabled": bool(r["enabled"]),
        })
    return plans


def _list_targets(conn: sqlite3.Connection, project_id: int) -> list[dict[str, Any]]:
    sql = (
        "SELECT Id, name, active, ra, dec, epochcode, rotation, roi, projectid "
        "FROM target WHERE projectid = ? AND active = 1 ORDER BY Id ASC"
    )
    return [
        {
            "id": r["Id"],
            "name": r["name"],
            "ra": r["ra"],
            "dec": r["dec"],
            "epoch_code": r["epochcode"],
            "rotation": r["rotation"],
            "roi": r["roi"],
            "project_id": r["projectid"],
        }
        for r in conn.execute(sql, (project_id,))
    ]


def next_target(
    conn: sqlite3.Connection,
    *,
    profile_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Pick the next target with remaining frames to acquire.

    Walks active projects by priority desc, then active targets, returns the
    first whose exposure plan still has at least one enabled row with
    remaining > 0. Returns None if nothing eligible.

    This is the *simple* selection used by Phase 1 — no altitude, moon, or
    weather scoring. The Planner agent layers that intelligence on top.
    """
    for project in list_projects(conn, profile_id=profile_id, active_only=True):
        for target in _list_targets(conn, project["id"]):
            plans = get_exposure_plan(conn, target_id=target["id"])
            actionable = [p for p in plans if p["enabled"] and p["remaining"] > 0]
            if actionable:
                return {
                    "project": project,
                    "target": target,
                    "plans": actionable,
                }
    return None


# ---------------------------------------------------------------------------
# MCP-tool-shaped wrappers: take simple kwargs, return {Success,...,Type} dict
# ---------------------------------------------------------------------------

_ENVELOPE_TYPE = "TARGET_SCHEDULER_DB"
_ERROR_TYPE = "TargetSchedulerDBError"


def _ok(message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"Success": True, "Message": message, "Details": details, "Type": _ENVELOPE_TYPE}


def _err(exc: Exception) -> dict[str, Any]:
    return {
        "Success": False,
        "Error": str(exc),
        "ErrorType": _ERROR_TYPE,
        "ErrorDetails": {},
        "StatusCode": 500,
        "Type": _ENVELOPE_TYPE,
    }


def tool_list_projects(
    *,
    profile_id: Optional[str] = None,
    active_only: bool = True,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    try:
        conn = connect(db_path)
        try:
            projects = list_projects(conn, profile_id=profile_id, active_only=active_only)
        finally:
            conn.close()
        return _ok(
            f"Found {len(projects)} project(s) in Target Scheduler",
            {"Projects": projects},
        )
    except Exception as e:
        return _err(e)


def tool_get_exposure_plan(
    *,
    target_id: int,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    try:
        conn = connect(db_path)
        try:
            plans = get_exposure_plan(conn, target_id=target_id)
        finally:
            conn.close()
        return _ok(
            f"Found {len(plans)} exposure plan(s) for target {target_id}",
            {"TargetId": target_id, "Plans": plans},
        )
    except Exception as e:
        return _err(e)


def tool_next_target(
    *,
    profile_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    try:
        conn = connect(db_path)
        try:
            nt = next_target(conn, profile_id=profile_id)
        finally:
            conn.close()
        if nt is None:
            return _ok(
                "No actionable target in Target Scheduler",
                {"Found": False, "Project": None, "Target": None, "Plans": []},
            )
        return _ok(
            f"Next target: {nt['target']['name']} (project: {nt['project']['name']})",
            {
                "Found": True,
                "Project": nt["project"],
                "Target": nt["target"],
                "Plans": nt["plans"],
            },
        )
    except Exception as e:
        return _err(e)
