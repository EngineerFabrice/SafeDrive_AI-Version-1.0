"""Driver -> Umusare assistance workflow: authorization, matching, state machine, privacy, audit."""
import json
import threading

import pytest

from tests.web.conftest import login
from website import assistance_service as svc
from website.geo import Candidate, approximate, haversine_km, rank_candidates

pytestmark = pytest.mark.db

KIGALI = (-1.944100, 30.061900)          # driver position used throughout
KM_LAT = 1 / 111.2                       # degrees of latitude per km (approx.)


def north(km):
    return KIGALI[0] + km * KM_LAT, KIGALI[1]


@pytest.fixture
def world(db):
    """Two cooperatives, one driver in coop A, helpers to add Umusare."""
    a, b = db.cooperative("Coop A", "A"), db.cooperative("Coop B", "B")
    driver = db.user("driver@example.com", role="driver", coop_id=a)

    def umusare(email, coop, km, verification="VERIFIED", availability="AVAILABLE"):
        uid = db.user(email, role="umusare", coop_id=coop)
        db.query("INSERT INTO umusare_profiles (user_id, verification_status, availability) VALUES (%s,%s,%s)",
                 (uid, verification, availability))
        lat, lon = north(km)
        db.query("INSERT INTO user_locations (user_id, lat, lon, updated_at) VALUES (%s,%s,%s,UTC_TIMESTAMP(3))",
                 (uid, lat, lon))
        return uid

    return {"a": a, "b": b, "driver": driver, "umusare": umusare, "db": db}


def post(client, url, body=None):
    return client.post(url, data=json.dumps(body or {}), content_type="application/json")


def request_help(client, lat=KIGALI[0], lon=KIGALI[1], trigger="SAFETY_ALERT"):
    return post(client, "/assistance/requests", {"lat": lat, "lon": lon, "accuracy": 12, "trigger": trigger})


def actions(db):
    return [r["action"] for r in db.query("SELECT action FROM audit_logs ORDER BY id")]


# ---------------------------------------------------------------- pure ranking (no database needed)
def test_ranking_distance_dominates_and_same_cooperative_is_only_a_bonus():
    near_other = Candidate(1, *north(1.2), cooperative_id=2)
    far_same = Candidate(2, *north(3.0), cooperative_id=1)
    assert [r.candidate.umusare_id for r in rank_candidates(*KIGALI, 1, [far_same, near_other], 5)] == [1, 2]
    close_same = Candidate(3, *north(1.4), cooperative_id=1)          # near-tie: the bonus decides
    assert rank_candidates(*KIGALI, 1, [near_other, close_same], 5)[0].candidate.umusare_id == 3


def test_ranking_respects_radius_and_reliability():
    reliable = Candidate(1, *north(2.0), 2, requests_received=10, requests_accepted=10)
    flaky = Candidate(2, *north(1.8), 2, requests_received=10, requests_accepted=0)
    far = Candidate(3, *north(8.0), 2)
    ranked = rank_candidates(*KIGALI, None, [reliable, flaky, far], 5)
    assert [r.candidate.umusare_id for r in ranked] == [1, 2]        # far one outside radius
    assert haversine_km(*KIGALI, *north(10)) == pytest.approx(10, rel=0.01)


def test_state_machine_rules():
    svc.check_transition("REQUESTED", "MATCHING")
    svc.check_transition("MATCHING", "ACCEPTED")
    svc.check_transition("ACCEPTED", "DRIVER_CONNECTED")
    svc.check_transition("DRIVER_CONNECTED", "COMPLETED")
    for current, new in [("MATCHING", "COMPLETED"), ("ACCEPTED", "COMPLETED"), ("DRIVER_CONNECTED", "CANCELLED"),
                         ("COMPLETED", "MATCHING"), ("CANCELLED", "ACCEPTED"), ("NO_UMUSARE_AVAILABLE", "MATCHING"),
                         ("REQUESTED", "ACCEPTED")]:
        with pytest.raises(svc.InvalidTransition):
            svc.check_transition(current, new)


# ---------------------------------------------------------------- creating requests
def test_driver_creates_request_and_nearby_umusare_is_offered(client, world):
    u = world["umusare"]("u1@example.com", world["a"], 1.0)
    login(client, "driver@example.com")
    r = request_help(client)
    assert r.status_code == 201
    body = r.get_json()
    assert body["status"] == "MATCHING" and body["umusare_contacted"] == 1 and "umusare" not in body
    offer = world["db"].query("SELECT umusare_id, status FROM assistance_offers")[0]
    assert (offer["umusare_id"], offer["status"]) == (u, "OFFERED")
    req = world["db"].query("SELECT trigger_source FROM assistance_requests")[0]
    assert req["trigger_source"] == "DRIVER_INITIATED"   # browser claimed SAFETY_ALERT, no live session confirms it
    assert actions(world["db"])[-2:] == ["ASSISTANCE_REQUESTED", "ASSISTANCE_MATCHING"]


def test_unauthenticated_cannot_request(client, world):
    assert request_help(client).status_code in (302, 401)
    assert world["db"].query("SELECT COUNT(*) AS n FROM assistance_requests")[0]["n"] == 0


@pytest.mark.parametrize("role", ["umusare", "manager", "admin"])
def test_non_driver_cannot_request(client, world, role):
    world["db"].user("x@example.com", role=role, coop_id=world["a"])
    login(client, "x@example.com")
    assert request_help(client).status_code == 403


def test_duplicate_active_requests_prevented(client, world):
    world["umusare"]("u1@example.com", world["a"], 1.0)
    login(client, "driver@example.com")
    assert request_help(client).status_code == 201
    second = request_help(client)
    assert second.status_code == 409 and second.get_json()["code"] == "ACTIVE_REQUEST_EXISTS"
    assert world["db"].query("SELECT COUNT(*) AS n FROM assistance_requests")[0]["n"] == 1


def test_concurrent_duplicate_clicks_create_one_request(app, world):
    world["umusare"]("u1@example.com", world["a"], 1.0)
    results, barrier = [], threading.Barrier(2)

    def click():
        c = app.test_client()
        login(c, "driver@example.com")
        barrier.wait()
        results.append(request_help(c).status_code)
    threads = [threading.Thread(target=click) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [201, 409]


@pytest.mark.parametrize("lat, lon", [(None, None), (95, 30), ("abc", 1), (0, 0)])
def test_invalid_location_rejected(client, world, lat, lon):
    login(client, "driver@example.com")
    r = request_help(client, lat, lon)
    assert r.status_code == 400 and r.get_json()["code"] == "INVALID_LOCATION"


# ---------------------------------------------------------------- matching
def test_no_eligible_umusare_gives_clear_status_and_fallback(client, world):
    world["db"].user("mgr@example.com", role="manager", coop_id=world["a"])
    world["umusare"]("pending@example.com", world["a"], 1.0, verification="PENDING")
    world["umusare"]("offline@example.com", world["a"], 1.0, availability="OFFLINE")
    world["umusare"]("toofar@example.com", world["a"], 80.0)
    login(client, "driver@example.com")
    body = request_help(client).get_json()
    assert body["status"] == "NO_UMUSARE_AVAILABLE"
    assert [c["name"] for c in body["fallback_contacts"]] == ["mgr"]
    row = world["db"].query("SELECT pickup_lat, pickup_lon, ended_at FROM assistance_requests")[0]
    assert row["pickup_lat"] is None and row["ended_at"] is not None     # exact point erased
    assert "ASSISTANCE_NO_UMUSARE_AVAILABLE" in actions(world["db"])
    assert request_help(client).status_code == 201                       # final state: may request again


def test_search_radius_expands(client, world):
    world["umusare"]("u10@example.com", world["a"], 10.0)
    login(client, "driver@example.com")
    body = request_help(client).get_json()
    assert body["status"] == "MATCHING" and body["search_radius_km"] == 15


def test_other_cooperative_umusare_is_matched(client, world):
    u = world["umusare"]("ub@example.com", world["b"], 2.0)
    login(client, "driver@example.com")
    request_help(client)
    assert world["db"].query("SELECT umusare_id, same_cooperative FROM assistance_offers")[0] == \
        {"umusare_id": u, "same_cooperative": 0}


def test_much_closer_other_cooperative_outranks_same_cooperative(client, world, monkeypatch):
    monkeypatch.setenv("ASSISTANCE_OFFERS_PER_ROUND", "1")
    world["umusare"]("same@example.com", world["a"], 3.0)
    other = world["umusare"]("other@example.com", world["b"], 1.2)
    login(client, "driver@example.com")
    request_help(client)
    assert [r["umusare_id"] for r in world["db"].query("SELECT umusare_id FROM assistance_offers")] == [other]


def test_same_cooperative_bonus_breaks_near_ties(client, world, monkeypatch):
    monkeypatch.setenv("ASSISTANCE_OFFERS_PER_ROUND", "1")
    same = world["umusare"]("same@example.com", world["a"], 1.4)
    world["umusare"]("other@example.com", world["b"], 1.2)
    login(client, "driver@example.com")
    request_help(client)
    assert [r["umusare_id"] for r in world["db"].query("SELECT umusare_id FROM assistance_offers")] == [same]


def test_decline_moves_to_next_candidate_then_no_umusare(client, app, world, monkeypatch):
    monkeypatch.setenv("ASSISTANCE_OFFERS_PER_ROUND", "1")
    world["umusare"]("u1@example.com", world["a"], 1.0)
    world["umusare"]("u2@example.com", world["b"], 2.0)
    login(client, "driver@example.com")
    req_id = request_help(client).get_json()["id"]
    for email in ("u1@example.com", "u2@example.com"):
        c = app.test_client()
        login(c, email)
        assert post(c, f"/assistance/requests/{req_id}/decline").status_code == 200
    assert client.get("/assistance/requests/current").get_json()["request"]["status"] == "NO_UMUSARE_AVAILABLE"
    assert actions(world["db"]).count("ASSISTANCE_DECLINED") == 2


def test_expired_offer_triggers_next_round(client, world, monkeypatch):
    monkeypatch.setenv("ASSISTANCE_OFFERS_PER_ROUND", "1")
    first = world["umusare"]("u1@example.com", world["a"], 1.0)
    second = world["umusare"]("u2@example.com", world["a"], 2.0)
    login(client, "driver@example.com")
    request_help(client)
    world["db"].query("UPDATE assistance_offers SET offered_at = UTC_TIMESTAMP(3) - INTERVAL 10 MINUTE")
    view = client.get("/assistance/requests/current").get_json()["request"]
    assert view["status"] == "MATCHING" and view["matching_round"] == 2
    offers = {r["umusare_id"]: r["status"] for r in world["db"].query("SELECT umusare_id, status FROM assistance_offers")}
    assert offers == {first: "EXPIRED", second: "OFFERED"}


# ---------------------------------------------------------------- accepting
def _offered_request(client, app, world, *umusare_emails):
    for i, email in enumerate(umusare_emails):
        world["umusare"](email, world["a"], 1.0 + i)
    login(client, "driver@example.com")
    req_id = request_help(client).get_json()["id"]
    clients = []
    for email in umusare_emails:
        c = app.test_client()
        login(c, email)
        clients.append(c)
    return req_id, clients


def test_eligible_umusare_accepts(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    r = post(u, f"/assistance/requests/{req_id}/accept")
    assert r.status_code == 200 and r.get_json()["active"]["id"] == req_id
    view = client.get("/assistance/requests/current").get_json()["request"]
    assert view["status"] == "ACCEPTED" and view["umusare"]["name"] == "u1" and view["umusare"]["verified"]
    assert view["umusare"]["cooperative"] == "Coop A" and view["umusare"]["distance_km"] is not None
    assert world["db"].query("SELECT availability FROM umusare_profiles")[0]["availability"] == "BUSY"
    assert "ASSISTANCE_ACCEPTED" in actions(world["db"])


def test_two_umusare_cannot_both_accept(client, app, world):
    req_id, clients = _offered_request(client, app, world, "u1@example.com", "u2@example.com")
    results, barrier = [], threading.Barrier(2)

    def accept(c):
        barrier.wait()
        results.append(post(c, f"/assistance/requests/{req_id}/accept").status_code)
    threads = [threading.Thread(target=accept, args=(c,)) for c in clients]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [200, 409]
    assert world["db"].query("SELECT COUNT(*) AS n FROM assistance_offers WHERE status='ACCEPTED'")[0]["n"] == 1


def test_unverified_umusare_cannot_accept(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    world["db"].query("UPDATE umusare_profiles SET verification_status='SUSPENDED'")
    r = post(u, f"/assistance/requests/{req_id}/accept")
    assert r.status_code == 403 and r.get_json()["code"] == "NOT_VERIFIED"


def test_unavailable_umusare_cannot_accept(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    assert post(u, "/assistance/umusare/availability", {"available": False}).get_json()["availability"] == "OFFLINE"
    r = post(u, f"/assistance/requests/{req_id}/accept")
    assert r.status_code == 409
    assert world["db"].query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id IN "
                             "(SELECT user_id FROM umusare_profiles)")[0]["n"] == 0   # offline: position deleted


def test_umusare_not_offered_cannot_see_or_accept(client, app, world, monkeypatch):
    monkeypatch.setenv("ASSISTANCE_OFFERS_PER_ROUND", "1")
    req_id, (u1, u2) = _offered_request(client, app, world, "u1@example.com", "u2@example.com")
    assert u2.get("/assistance/umusare/status").get_json()["incoming"] == []
    assert post(u2, f"/assistance/requests/{req_id}/accept").status_code == 404
    assert post(u2, f"/assistance/requests/{req_id}/location", {"lat": 1, "lon": 1}).status_code == 404


def test_cancelled_or_accepted_request_cannot_be_accepted(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    assert post(client, f"/assistance/requests/{req_id}/cancel").get_json()["status"] == "CANCELLED"
    assert post(u, f"/assistance/requests/{req_id}/accept").status_code == 409


# ---------------------------------------------------------------- ownership
def test_driver_cannot_touch_another_drivers_request(client, app, world):
    req_id, _ = _offered_request(client, app, world, "u1@example.com")
    world["db"].user("other@example.com", role="driver", coop_id=world["a"])
    other = app.test_client()
    login(other, "other@example.com")
    assert post(other, f"/assistance/requests/{req_id}/cancel").status_code == 404
    assert post(other, f"/assistance/requests/{req_id}/location", {"lat": 1, "lon": 1}).status_code == 404
    assert other.get("/assistance/requests/current").get_json()["request"] is None


def test_umusare_endpoints_reject_drivers(client, world):
    login(client, "driver@example.com")
    assert client.get("/assistance/umusare/status").status_code == 403
    assert post(client, "/assistance/requests/1/accept").status_code == 403


# ---------------------------------------------------------------- privacy and location sharing
def test_exact_location_hidden_before_acceptance(client, app, world):
    lat, lon = -1.944137, 30.061982
    world["umusare"]("u1@example.com", world["a"], 1.0)
    login(client, "driver@example.com")
    req_id = request_help(client, lat, lon).get_json()["id"]
    u = app.test_client()
    login(u, "u1@example.com")
    raw = u.get("/assistance/umusare/status").get_data(as_text=True)
    status = json.loads(raw)
    offer = status["incoming"][0]
    assert offer["approx_area"] == dict(zip(("lat", "lon"), approximate(lat, lon)))
    assert offer["approx_distance"] in ("less than 1 km", "about 1.0 km", "about 1.5 km")
    assert "-1.944137" not in raw and "30.061982" not in raw and "driver" not in json.dumps(offer)
    assert status["active"] is None
    # live location sharing is not possible before acceptance
    assert post(client, f"/assistance/requests/{req_id}/location", {"lat": lat, "lon": lon}).status_code == 409


def test_locations_shared_after_acceptance_and_stopped_after_completion(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    post(u, f"/assistance/requests/{req_id}/accept")
    active = u.get("/assistance/umusare/status").get_json()["active"]
    assert active["pickup"] == {"lat": KIGALI[0], "lon": KIGALI[1]} and active["driver"]["name"] == "driver"
    assert post(client, f"/assistance/requests/{req_id}/location", {"lat": -1.95, "lon": 30.06}).status_code == 200
    assert u.get("/assistance/umusare/status").get_json()["active"]["driver_location"]["lat"] == -1.95

    assert post(u, f"/assistance/requests/{req_id}/complete").status_code == 409        # must connect first
    assert post(client, f"/assistance/requests/{req_id}/connect").get_json()["status"] == "DRIVER_CONNECTED"
    assert post(client, f"/assistance/requests/{req_id}/cancel").status_code == 409     # too late to cancel
    assert post(u, f"/assistance/requests/{req_id}/complete").get_json()["status"] == "COMPLETED"

    db = world["db"]
    assert post(client, f"/assistance/requests/{req_id}/location", {"lat": -1.95, "lon": 30.06}).status_code == 409
    assert db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (world["driver"],))[0]["n"] == 0
    assert db.query("SELECT pickup_lat FROM assistance_requests")[0]["pickup_lat"] is None
    assert db.query("SELECT availability FROM umusare_profiles")[0]["availability"] == "AVAILABLE"
    view = client.get("/assistance/requests/current").get_json()["request"]
    assert view["status"] == "COMPLETED" and "umusare" not in view and not view["sharing"]
    assert [a for a in actions(db) if a.startswith("ASSISTANCE_")] == [
        "ASSISTANCE_REQUESTED", "ASSISTANCE_MATCHING", "ASSISTANCE_ACCEPTED", "ASSISTANCE_CONNECTED",
        "ASSISTANCE_COMPLETED"]
    details = " ".join(str(r["details"]) for r in db.query("SELECT details FROM audit_logs"))
    assert "30.06" not in details and "-1.9" not in details                 # no coordinates in the audit log


def test_cancel_after_acceptance_stops_tracking(client, app, world):
    req_id, (u,) = _offered_request(client, app, world, "u1@example.com")
    post(u, f"/assistance/requests/{req_id}/accept")
    post(client, f"/assistance/requests/{req_id}/location", {"lat": -1.95, "lon": 30.06})
    assert post(client, f"/assistance/requests/{req_id}/cancel").get_json()["status"] == "CANCELLED"
    assert post(u, f"/assistance/requests/{req_id}/location", {"lat": -1.95, "lon": 30.06}).status_code == 409
    assert world["db"].query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s",
                             (world["driver"],))[0]["n"] == 0
    assert "ASSISTANCE_CANCELLED" in actions(world["db"])


def test_unverified_umusare_cannot_go_available(client, world):
    world["db"].user("p@example.com", role="umusare", coop_id=world["a"])
    world["db"].query("INSERT INTO umusare_profiles (user_id) SELECT id FROM users WHERE email='p@example.com'")
    login(client, "p@example.com")
    r = post(client, "/assistance/umusare/availability", {"available": True, "lat": -1.9, "lon": 30.0})
    assert r.status_code == 403 and r.get_json()["code"] == "NOT_VERIFIED"


def test_assistance_posts_require_csrf(world):
    from website import create_app
    c = create_app({"BCRYPT_LOG_ROUNDS": 4}).test_client()        # CSRF enabled
    assert post(c, "/assistance/requests", {"lat": 1, "lon": 1}).status_code == 400
