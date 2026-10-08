# website/migrate.py
"""Minimal, non-destructive SQL migrations for the MySQL database.

Applies `website/migrations/NNNN_*.sql` files in order, once each, and records
them in a `schema_migrations` table. Migration files must be additive (CREATE
TABLE IF NOT EXISTS, ADD COLUMN, ...); this tool never drops or resets anything.

    python -m website.migrate                    # apply pending migrations
    python -m website.migrate --status           # list pending migrations
    python -m website.migrate --create-database  # CREATE DATABASE IF NOT EXISTS first

Uses the same connection settings as the app (see website/config.py; the
default database is safedrive_ai_v2).
"""
import argparse
import glob
import os
import re

import pymysql

from . import get_connection
from .config import db_settings

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")
# Migrations only ever run on the v2 database or its suffixed copies (e.g. safedrive_ai_v2_test).
_DB_NAME = re.compile(r"^safedrive_ai_v2(_[a-z0-9_]{1,40})?$")


def target_database():
    """The configured database name, refused unless it is a v2 database (never the legacy one)."""
    name = db_settings()["database"]          # raises ConfigError for the legacy database
    if not _DB_NAME.match(name):
        raise ValueError(f"refusing to migrate {name!r}: only safedrive_ai_v2 or safedrive_ai_v2_* are allowed")
    return name


_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(100) NOT NULL PRIMARY KEY,
    applied_at  DATETIME(3)  NOT NULL DEFAULT CURRENT_TIMESTAMP(3)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


def _statements(sql):
    sql = re.sub(r"--[^\n]*", "", sql)              # strip line comments
    return [s.strip() for s in sql.split(";") if s.strip()]


def migration_files():
    return sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "[0-9][0-9][0-9][0-9]_*.sql")))


def applied_versions(cursor):
    cursor.execute(_TRACKING_TABLE)
    cursor.execute("SELECT version FROM schema_migrations")
    return {row["version"] for row in cursor.fetchall()}


def create_database():
    """CREATE DATABASE IF NOT EXISTS for the configured name; never alters an existing one."""
    name = target_database()
    settings = db_settings()
    settings.pop("database")
    conn = pymysql.connect(**settings, connect_timeout=10)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{name}` "
                           "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        conn.commit()
    finally:
        conn.close()
    return name


def migrate(dry_run=False):
    """Apply pending migrations; returns the list of versions applied (or pending if dry_run)."""
    target_database()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        done = applied_versions(cursor)
        pending = [f for f in migration_files() if os.path.basename(f) not in done]
        if dry_run:
            return [os.path.basename(f) for f in pending]
        applied = []
        for path in pending:
            version = os.path.basename(path)
            with open(path, encoding="utf-8") as fh:
                for statement in _statements(fh.read()):
                    cursor.execute(statement)
            cursor.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            conn.commit()
            applied.append(version)
        return applied
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply SafeDrive AI database migrations")
    parser.add_argument("--status", action="store_true", help="only list pending migrations")
    parser.add_argument("--create-database", action="store_true",
                        help="create the configured database first if it does not exist")
    args = parser.parse_args()
    if args.create_database:
        print(f"database: {create_database()}")
    result = migrate(dry_run=args.status)
    label = "pending" if args.status else "applied"
    print(f"{label}: {', '.join(result) if result else 'none'}")
