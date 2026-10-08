"""Nearby verified drivers: eligibility filtering, opt-in, and location privacy (MySQL test database)."""
import json

import pytest

from tests.web.conftest import login

pytestmark = pytest.mark.db

HERE = (-1.944123, 30.061987)                   # exact positions must never come back out
CLOSE = (-1.951234, 30.068765)                  # ~1 km away
FAR = (-1.700000, 30.300000)                    # > 30 km away


def post(client, url, body):
    return client.post(url, data=json.dumps(body), content_type="application/json")


class Town:
    def __init__(self, app, db):
        self.app, self.db = app, db
        self.coop = db.cooperative("Coop A", "CA")
        self.other_coop = db.cooperative("Coop B", "CB")
        self.viewer = self.driver("viewer@example.com")
        self.clients = {}

    def driver(self, email, coop=None, status="VERIFIED", plate="RAB 123 A", visible=True):
        uid = self.db.user(email, coop_id=coop or self.coop)
        self.db.query("INSERT INTO driver_profiles (user_id, verification_status, vehicle_plate_number, nearby_visibility) "
                      "VALUES (%s,%s,%s,%s)", (uid, status, plate, int(visible)))
        return uid

    def c(self, email):
        if email not in self.clients:
            self.clients[email] = self.app.test_client()
            login(self.clients[email], email)
        return self.clients[email]

    def share(self, email, pos):
        """Propose a position (the server keeps only the ~1 km cell, if plausible)."""
        return post(self.c(email), "/driver/presence", {"lat": pos[0], "lon": pos[1]})

    def look(self, pos=HERE, email="viewer@example.com"):
        r = self.share(email, pos)
        assert r.status_code in (200, 429), r.get_json()        # 429: too soon after the last update, old cell kept
        r = post(self.c(email), "/driver/nearby", {})
        assert r.status_code == 200, r.get_json()
        return r.get_json()

    def names(self, data):
        return {d["name"] for c in data["cells"] for d in c["drivers"]}


@pytest.fixture
def t(app, db):
    return Town(app, db)


def test_verified_opted_in_driver_appears_with_safe_information_only(t):
    t.driver("near@example.com")
    t.db.query("UPDATE users SET phone='+250788999000' WHERE email='near@example.com'")
    t.share("near@example.com", CLOSE)
    data = t.look()
    assert data["total"] == 1 and t.names(data) == {"near"}
    cell = data["cells"][0]
    d = cell["drivers"][0]
    assert set(d) == {"name", "cooperative", "group", "verified"}       # no id, phone, email or plate
    raw = json.dumps(data)
    for secret in ("-1.951234", "30.068765", "-1.944123", "30.061987", "+250788999000", "near@example.com", "RAB"):
        assert secret not in raw, secret
    assert round(cell["lat"], 2) == cell["lat"] and round(cell["lon"], 2) == cell["lon"]   # ~1 km grid only
    assert cell["distance_label"] in ("Within ~1 km", "About 2 km away", "About 1 km away")


def test_only_the_grid_cell_is_stored(t):
    t.look()
    row = t.db.query("SELECT approx_lat, approx_lon FROM driver_presence WHERE user_id=%s", (t.viewer,))[0]
    assert (float(row["approx_lat"]), float(row["approx_lon"])) == (-1.94, 30.06)


@pytest.mark.parametrize("label, setup", [
    ("unverified", lambda t: t.driver("x@example.com", status="PENDING")),
    ("suspended", lambda t: t.driver("x@example.com", status="SUSPENDED")),
    ("rejected", lambda t: t.driver("x@example.com", status="REJECTED")),
    ("no plate", lambda t: t.driver("x@example.com", plate=None)),
    ("opted out", lambda t: t.driver("x@example.com", visible=False)),
])
def test_ineligible_drivers_are_not_shown(t, label, setup):
    uid = setup(t)
    t.db.query("INSERT INTO driver_presence (user_id, approx_lat, approx_lon, updated_at) VALUES (%s,-1.95,30.07,UTC_TIMESTAMP(3))",
               (uid,))
    assert t.look()["total"] == 0, label


def test_deactivated_unverified_email_and_suspended_cooperative_are_not_shown(t):
    a = t.driver("a@example.com"); b = t.driver("b@example.com"); c = t.driver("c@example.com", coop=t.other_coop)
    for uid in (a, b, c):
        t.db.query("INSERT INTO driver_presence (user_id, approx_lat, approx_lon, updated_at) VALUES "
                   "(%s,-1.95,30.07,UTC_TIMESTAMP(3))", (uid,))
    t.db.query("UPDATE users SET is_active=0 WHERE id=%s", (a,))
    t.db.query("UPDATE users SET email_verified_at=NULL WHERE id=%s", (b,))
    t.db.query("UPDATE cooperatives SET status='SUSPENDED' WHERE id=%s", (t.other_coop,))
    assert t.look()["total"] == 0


def test_stale_and_distant_presence_is_not_shown(t):
    stale, far = t.driver("stale@example.com"), t.driver("far@example.com")
    t.db.query("INSERT INTO driver_presence (user_id, approx_lat, approx_lon, updated_at) VALUES "
               "(%s,-1.95,30.07,UTC_TIMESTAMP(3) - INTERVAL 10 MINUTE)", (stale,))
    t.share("far@example.com", FAR)
    assert t.look()["total"] == 0


def test_other_cooperatives_verified_drivers_are_shown_with_their_cooperative(t):
    t.driver("other@example.com", coop=t.other_coop)
    t.share("other@example.com", CLOSE)
    data = t.look()
    assert data["cells"][0]["drivers"][0]["cooperative"] == "Coop B"


def test_drivers_in_the_same_cell_are_aggregated(t):
    for i in range(3):
        t.driver(f"n{i}@example.com")
        t.share(f"n{i}@example.com", CLOSE)
    data = t.look()
    assert data["total"] == 3 and len(data["cells"]) == 1 and data["cells"][0]["count"] == 3


def test_viewer_never_sees_themself(t):
    t.look()
    assert t.look()["total"] == 0


def test_unverified_viewer_is_refused(t):
    t.driver("pending@example.com", status="PENDING")
    r = t.share("pending@example.com", HERE)
    assert r.status_code == 403 and "verified" in r.get_json()["error"]
    assert t.db.query("SELECT COUNT(*) AS n FROM driver_presence")[0]["n"] == 0


def test_opting_out_removes_presence_immediately(t):
    t.driver("near@example.com")
    t.share("near@example.com", CLOSE)
    assert t.look()["total"] == 1
    r = post(t.c("near@example.com"), "/driver/nearby/visibility", {"visible": False})
    assert r.get_json() == {"visible": False}
    assert t.look()["total"] == 0                                         # hidden immediately
    rows = t.db.query("SELECT approx_lat, approx_lon FROM driver_presence WHERE user_id=(SELECT id FROM users "
                      "WHERE email='near@example.com')")
    assert all(round(float(r["approx_lat"]), 2) == float(r["approx_lat"]) for r in rows)   # only the ~1 km cell
    t.share("near@example.com", CLOSE)                                    # opted out: viewing does not re-publish
    assert t.look()["total"] == 0


def test_invalid_coordinates_are_rejected(t):
    assert t.share("viewer@example.com", (200, 30)).status_code == 400   # /driver/presence validates updates


def test_nearby_requires_csrf(db):
    from website import create_app
    app = create_app({"WTF_CSRF_ENABLED": True, "BCRYPT_LOG_ROUNDS": 4, "TESTING": True})
    t = Town(app, db)
    c = app.test_client()
    token = c.get("/login").get_data(as_text=True).split('name="csrf_token" value="')[1].split('"')[0]
    c.post("/login", data={"email": "viewer@example.com", "password": "Passw0rd!", "csrf_token": token})
    assert post(c, "/driver/nearby", {"lat": HERE[0], "lon": HERE[1]}).status_code == 400


def test_dashboard_never_requests_location_on_load_and_explains_states(t):
    html = t.c("viewer@example.com").get("/driver-dashboard").get_data(as_text=True)
    assert 'data-eligible="1"' in html and "js/nearby.js" in html
    js = t.c("viewer@example.com").get("/static/js/nearby.js").get_data(as_text=True)
    assert "Never prompt on page load" in js and 'permissions.query({ name: "geolocation" })' in js
    for text in ("No verified drivers are currently visible nearby.", "Location unavailable. Enable location to view nearby drivers.",
                 "Try again", "You are offline"):
        assert text in js, text
    t.driver("pending@example.com", status="PENDING")
    assert 'data-eligible="0"' in t.c("pending@example.com").get("/driver-dashboard").get_data(as_text=True)
