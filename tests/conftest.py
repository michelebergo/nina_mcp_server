"""Shared test fixtures for Target Scheduler DB tests.

Builds an in-memory SQLite database mirroring the real Target Scheduler v5
schema (extracted live from %LOCALAPPDATA%\\NINA\\SchedulerPlugin\\schedulerdb.sqlite).
Tests never touch the user's real DB.
"""

import sqlite3
import tempfile
from pathlib import Path
import pytest


TS_SCHEMA = """
CREATE TABLE project (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    state INTEGER,
    priority INTEGER,
    createdate INTEGER,
    activedate INTEGER,
    inactivedate INTEGER,
    minimumtime INTEGER,
    minimumaltitude REAL,
    usecustomhorizon INTEGER,
    horizonoffset REAL,
    meridianwindow INTEGER,
    filterswitchfrequency INTEGER,
    ditherevery INTEGER,
    enablegrader INTEGER,
    isMosaic INTEGER NOT NULL DEFAULT 0,
    flatsHandling INTEGER NOT NULL DEFAULT 0,
    maximumAltitude REAL DEFAULT 0,
    smartexposureorder INTEGER DEFAULT 0,
    guid TEXT
);

CREATE TABLE target (
    Id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    active INTEGER NOT NULL,
    ra REAL,
    dec REAL,
    epochcode INTEGER NOT NULL,
    rotation REAL,
    roi REAL,
    projectid INTEGER,
    unusedOEO TEXT,
    guid TEXT,
    FOREIGN KEY(projectid) REFERENCES project(Id)
);

CREATE TABLE exposuretemplate (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL,
    name TEXT NOT NULL,
    filtername TEXT NOT NULL,
    gain INTEGER,
    offset INTEGER,
    bin INTEGER,
    readoutmode INTEGER,
    twilightlevel INTEGER,
    moonavoidanceenabled INTEGER,
    moonavoidanceseparation REAL,
    moonavoidancewidth INTEGER,
    maximumhumidity REAL,
    defaultexposure REAL DEFAULT 60,
    moonrelaxscale REAL DEFAULT 0,
    moonrelaxmaxaltitude REAL DEFAULT 5,
    moonrelaxminaltitude REAL DEFAULT -15,
    moondownenabled INTEGER DEFAULT 0,
    ditherevery INTEGER DEFAULT -1,
    minutesOffset INTEGER DEFAULT 0,
    guid TEXT
);

CREATE TABLE exposureplan (
    Id INTEGER NOT NULL PRIMARY KEY,
    profileId TEXT NOT NULL,
    exposure REAL NOT NULL,
    desired INTEGER,
    acquired INTEGER,
    accepted INTEGER,
    targetid INTEGER,
    exposureTemplateId INTEGER,
    enabled INTEGER DEFAULT 1,
    guid TEXT,
    FOREIGN KEY(targetid) REFERENCES target(Id),
    FOREIGN KEY(exposureTemplateId) REFERENCES exposuretemplate(Id)
);

CREATE TABLE acquiredimage (
    Id INTEGER NOT NULL PRIMARY KEY,
    projectId INTEGER NOT NULL,
    targetId INTEGER NOT NULL,
    acquireddate INTEGER,
    filtername TEXT NOT NULL,
    gradingStatus INTEGER NOT NULL,
    metadata TEXT NOT NULL,
    rejectreason TEXT,
    profileId TEXT,
    exposureId INTEGER DEFAULT 0,
    guid TEXT
);
"""


@pytest.fixture
def empty_db(tmp_path):
    """Path to a fresh on-disk SQLite DB with TS schema but zero rows."""
    db_path = tmp_path / "schedulerdb.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(TS_SCHEMA)
    conn.commit()
    conn.close()
    return db_path


PROFILE_A = "11111111-1111-1111-1111-111111111111"
PROFILE_B = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def seeded_db(empty_db):
    """DB with two profiles, three projects (1 active, 1 inactive, 1 draft),
    targets, and exposure plans — enough to exercise filtering/joins/selection."""
    conn = sqlite3.connect(str(empty_db))

    # State enum: 0=Draft, 1=Active, 2=Inactive (per TS plugin convention)
    conn.executemany(
        "INSERT INTO project (Id, profileId, name, state, priority, minimumaltitude, "
        "maximumAltitude, meridianwindow) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, PROFILE_A, "M81 Bode's Galaxy",    1, 100, 30.0, 0.0, 60),
            (2, PROFILE_A, "NGC 7000 N.America",   1, 50,  25.0, 0.0, 60),
            (3, PROFILE_A, "Drafty Project",       0, 200, 20.0, 0.0, 60),
            (4, PROFILE_B, "Other Rig Project",    1, 100, 30.0, 0.0, 60),
        ],
    )
    conn.executemany(
        "INSERT INTO target (Id, name, active, ra, dec, epochcode, projectid) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (10, "M81",       1, 148.888, 69.065, 2, 1),
            (11, "NGC 7000",  1, 314.750, 44.330, 2, 2),
            (12, "Disabled Target", 0, 0.0, 0.0, 2, 1),
            (13, "Draft Target",    1, 0.0, 0.0, 2, 3),
            (14, "Other Rig Target", 1, 0.0, 0.0, 2, 4),
        ],
    )
    conn.executemany(
        "INSERT INTO exposuretemplate (Id, profileId, name, filtername, gain, offset, bin) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (100, PROFILE_A, "L", "L", 100, 50, 1),
            (101, PROFILE_A, "Ha", "Ha", 100, 50, 1),
            (102, PROFILE_B, "L", "L", 0, 10, 1),
        ],
    )
    conn.executemany(
        "INSERT INTO exposureplan (Id, profileId, exposure, desired, acquired, accepted, "
        "targetid, exposureTemplateId, enabled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # M81: L plan still needs frames, Ha plan complete
            (1000, PROFILE_A, 180.0, 30, 10, 8,  10, 100, 1),
            (1001, PROFILE_A, 300.0, 20, 20, 18, 10, 101, 1),
            # NGC 7000: L plan has frames but desired > acquired; Ha plan disabled
            (1002, PROFILE_A, 120.0, 50, 5,  5,  11, 100, 1),
            (1003, PROFILE_A, 600.0, 10, 0,  0,  11, 101, 0),
            # Other rig
            (1004, PROFILE_B, 60.0,  100, 0, 0,  14, 102, 1),
        ],
    )
    conn.commit()
    conn.close()
    return empty_db
