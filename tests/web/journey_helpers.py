"""Shared helpers for journey / fare / payment tests (one driver, one verified Umusare)."""
import json

from tests.web.conftest import login

START = (-1.944100, 30.061900)
KM_LAT = 1 / 111.195          # degrees of latitude per km (haversine with R = 6371.0088 km)


def north(km, base=START):
    return base[0] + km * KM_LAT, base[1]


def post(client, url, body=None):
    return client.post(url, data=json.dumps(body or {}), content_type="application/json")


class Journey:
    """Drives one assistance request through the workflow with real HTTP calls."""

    def __init__(self, app, db, coop_driver=None, coop_umusare=None, umusare_phone="+250788123456"):
        self.app, self.db = app, db
        a = coop_driver or db.cooperative("Kigali Safe Transport", "KST")
        b = coop_umusare or a
        self.driver_id = db.user("driver@example.com", role="driver", coop_id=a)
        self.umusare_id = db.user("umusare@example.com", role="umusare", coop_id=b)
        db.query("UPDATE users SET phone=%s, username='Jean Claude M.' WHERE id=%s", (umusare_phone, self.umusare_id))
        db.query("INSERT INTO umusare_profiles (user_id, verification_status, availability, verified_at) "
                 "VALUES (%s,'VERIFIED','AVAILABLE',UTC_TIMESTAMP(3))", (self.umusare_id,))
        self.set_umusare_position(*north(1.0))
        self.d = app.test_client()
        login(self.d, "driver@example.com")
        self.u = app.test_client()
        login(self.u, "umusare@example.com")
        self.id = None

    def set_umusare_position(self, lat, lon):
        self.db.query("INSERT INTO user_locations (user_id, lat, lon, updated_at) VALUES (%s,%s,%s,UTC_TIMESTAMP(3)) "
                      "ON DUPLICATE KEY UPDATE lat=VALUES(lat), lon=VALUES(lon), updated_at=VALUES(updated_at)",
                      (self.umusare_id, lat, lon))

    def request(self, trigger="DRIVER_INITIATED"):
        r = post(self.d, "/assistance/requests", {"lat": START[0], "lon": START[1], "trigger": trigger})
        assert r.status_code == 201, r.get_json()
        self.id = r.get_json()["id"]
        return r.get_json()

    def accept(self):
        assert post(self.u, f"/assistance/requests/{self.id}/accept").status_code == 200

    def arrive(self):
        self.set_umusare_position(*START)                  # Umusare is now with the driver
        assert post(self.u, f"/assistance/requests/{self.id}/connect").status_code == 200

    def drive(self, km_steps, minutes_per_step=2):
        """Umusare's device reports positions along a straight road north, one step every few minutes."""
        travelled = 0.0
        for step in km_steps:
            travelled += step
            self.db.query("UPDATE assistance_requests SET journey_last_at = UTC_TIMESTAMP(3) - INTERVAL %s MINUTE "
                          "WHERE id=%s", (minutes_per_step, self.id))
            lat, lon = north(travelled)
            r = post(self.u, f"/assistance/requests/{self.id}/location", {"lat": lat, "lon": lon, "accuracy": 8})
            assert r.status_code == 200, r.get_json()
        return travelled

    def complete(self, body=None):
        return post(self.u, f"/assistance/requests/{self.id}/complete", body)

    def row(self):
        return self.db.query("SELECT * FROM assistance_requests WHERE id=%s", (self.id,))[0]

    def to_payment_pending(self, steps=(0.46,) * 10):
        self.request()
        self.accept()
        self.arrive()
        self.drive(steps)
        assert self.complete().status_code == 200
