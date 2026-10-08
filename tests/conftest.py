"""Shared test setup.

Tests never touch the application databases: everything that needs MySQL uses
SAFEDRIVE_TEST_DB_NAME (default safedrive_ai_v2_test), and the fixtures refuse
to run against a database whose name does not end in "_test".
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Must be set before `website` is imported (it loads .env, which never overrides these).
os.environ["SAFEDRIVE_ENV"] = "testing"
os.environ["SAFEDRIVE_DB_NAME"] = os.environ.get("SAFEDRIVE_TEST_DB_NAME", "safedrive_ai_v2_test")
os.environ.pop("SAFEDRIVE_DEBUG", None)
os.environ.setdefault("MODEL_PROVIDER", "none")
os.environ["CARTO_API_KEY"] = ""      # a developer's real key in .env must not leak into tests (dotenv never overrides)
# A developer's real SMTP account (e.g. Gmail) in .env must never be used by tests: blank every MAIL_* setting.
for _name in ("MAIL_HOST", "MAIL_PORT", "MAIL_USERNAME", "MAIL_PASSWORD", "MAIL_FROM", "MAIL_USE_TLS", "MAIL_USE_SSL"):
    os.environ[_name] = ""
os.environ["MAIL_BACKEND"] = "memory"
