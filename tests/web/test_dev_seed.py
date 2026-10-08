"""Development/demo account seed: roles, hashing, cooperative links, idempotency, login."""
import pytest

from tests.web.conftest import login
from website import bcrypt
from website.auth import validate_password
from website.dev_seed import DEMO_COOPERATIVE, DEV_ACCOUNTS, SeedRefused, seed_dev_accounts

pytestmark = pytest.mark.db

LANDING = {"admin": "/admin-dashboard", "manager": "/manager-dashboard", "driver": "/driver-dashboard",
           "umusare": "/umusare-dashboard"}


def test_demo_passwords_pass_the_normal_password_policy():
    assert {a["role"] for a in DEV_ACCOUNTS} == {"admin", "manager", "driver", "umusare"}
    assert all(validate_password(a["password"]) == [] for a in DEV_ACCOUNTS)


def test_seed_creates_four_roles_hashed_and_linked_to_demo_cooperative(db):
    result = seed_dev_accounts()
    assert result["database"].endswith("_test")
    assert [a["status"] for a in result["accounts"]] == ["created"] * 4

    users = {r["email"]: r for r in db.query("SELECT id, email, role, password_hash FROM users")}
    for acc in DEV_ACCOUNTS:
        u = users[acc["email"]]
        assert u["role"] == acc["role"]
        assert u["password_hash"].startswith("$2") and acc["password"] not in u["password_hash"]
        assert bcrypt.check_password_hash(u["password_hash"], acc["password"])

    coop = db.query("SELECT id, name, status FROM cooperatives WHERE code=%s", (DEMO_COOPERATIVE["code"],))[0]
    assert coop["status"] == "APPROVED"
    members = {r["user_id"]: r for r in db.query("SELECT * FROM cooperative_memberships")}
    for role in ("manager", "driver", "umusare"):
        m = members[users[f"{role}@safedrive.ai"]["id"]]
        assert (m["cooperative_id"], m["member_role"], m["status"]) == (coop["id"], role, "APPROVED")
    assert users["admin@safedrive.ai"]["id"] not in members           # admins are not cooperative members
    assert db.query("SELECT user_id FROM driver_profiles")[0]["user_id"] == users["driver@safedrive.ai"]["id"]
    umu = db.query("SELECT verification_status, availability, verified_by FROM umusare_profiles")[0]
    assert (umu["verification_status"], umu["availability"]) == ("VERIFIED", "OFFLINE")
    assert umu["verified_by"] == users["admin@safedrive.ai"]["id"]


def test_seed_is_idempotent(db):
    seed_dev_accounts()
    before = {t: db.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
              for t in ("users", "cooperatives", "cooperative_memberships", "driver_profiles", "umusare_profiles")}
    second = seed_dev_accounts()
    assert all(a["status"] == "exists (unchanged)" for a in second["accounts"])
    assert not second["cooperative"]["created"]
    after = {t: db.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in before}
    assert after == before


def test_existing_account_is_never_overwritten(db):
    coop = db.cooperative()
    db.user("driver@safedrive.ai", role="driver", password="MyOwnPass1!", coop_id=coop)
    original = db.query("SELECT username, password_hash FROM users WHERE email='driver@safedrive.ai'")[0]
    result = seed_dev_accounts()
    status = {a["email"]: a["status"] for a in result["accounts"]}
    assert status["driver@safedrive.ai"] == "exists (unchanged)"
    assert db.query("SELECT username, password_hash FROM users WHERE email='driver@safedrive.ai'")[0] == original
    # the existing driver keeps its own cooperative, not the demo one
    m = db.query("SELECT cooperative_id FROM cooperative_memberships m JOIN users u ON u.id = m.user_id "
                 "WHERE u.email='driver@safedrive.ai'")
    assert [r["cooperative_id"] for r in m] == [coop]


@pytest.mark.parametrize("acc", DEV_ACCOUNTS, ids=[a["role"] for a in DEV_ACCOUNTS])
def test_seeded_accounts_can_log_in_and_reach_their_dashboard(client, db, acc):
    seed_dev_accounts()
    r = login(client, acc["email"], acc["password"])
    assert r.status_code == 302 and r.headers["Location"].endswith(LANDING[acc["role"]])
    assert client.get(LANDING[acc["role"]]).status_code == 200
    client.post("/logout")
    assert login(client, acc["email"], "WrongPassword1!").status_code == 200   # wrong password rejected


def test_seed_refuses_production(db, monkeypatch):
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    with pytest.raises(SeedRefused):
        seed_dev_accounts()
    assert db.query("SELECT COUNT(*) AS n FROM users")[0]["n"] == 0
