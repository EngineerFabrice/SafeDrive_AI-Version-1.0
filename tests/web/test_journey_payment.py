"""Journey completion (server-side distance + fare) and the two-step payment confirmation."""
from decimal import ROUND_HALF_UP, Decimal

import pytest

from tests.web.conftest import login
from tests.web.journey_helpers import START, Journey, north, post
from website.geo import haversine_km

pytestmark = pytest.mark.db


@pytest.fixture
def j(app, db):
    db.query("INSERT IGNORE INTO pricing_settings (id) VALUES (1)")     # 500 RWF/km default
    return Journey(app, db)


def expected_fare(km, rate=500):
    return (Decimal(str(round(km, 2))) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------- journey completion
def test_complete_journey_calculates_distance_and_fare_server_side(j):
    j.request(); j.accept(); j.arrive()
    travelled = j.drive([0.46] * 10)                       # 4.6 km north
    r = j.complete()
    assert r.status_code == 200 and r.get_json()["status"] == "COMPLETED"
    assert r.get_json()["payment_status"] == "PAYMENT_PENDING"
    row = j.row()
    km = float(row["final_distance_km"])
    assert km == pytest.approx(travelled, abs=0.02) and km == pytest.approx(4.6, abs=0.02)
    assert row["final_fare"] == expected_fare(km)          # 4.6 km x 500 = about RWF 2,300
    assert abs(row["final_fare"] - 2300) <= 10
    assert (row["fare_currency"], row["fare_price_per_km"], row["fare_base_fee"]) == ("RWF", Decimal("500.00"), Decimal("0.00"))
    assert row["payment_status"] == "PAYMENT_PENDING" and row["payment_phone"] == "+250788123456"
    assert row["journey_ended_at"] is not None


def test_browser_values_for_distance_fare_rate_and_status_are_ignored(j):
    j.request(); j.accept(); j.arrive()
    j.drive([1.0, 1.0])
    r = j.complete({"distance_km": 0.1, "final_fare": 1, "fare": 1, "price_per_km": 1,
                    "payment_status": "PAYMENT_COMPLETED", "phone": "+250700000000"})
    row = j.row()
    assert float(row["final_distance_km"]) == pytest.approx(2.0, abs=0.02)
    assert row["final_fare"] == expected_fare(float(row["final_distance_km"]))
    assert row["payment_status"] == "PAYMENT_PENDING" and row["payment_phone"] == "+250788123456"
    assert r.get_json()["fare"] == int(row["final_fare"])


def test_gps_jump_is_not_counted(j):
    j.request(); j.accept(); j.arrive()
    j.drive([1.0])
    # 50 km "in 2 minutes" (1500 km/h): a GPS glitch, not travel
    j.db.query("UPDATE assistance_requests SET journey_last_at = UTC_TIMESTAMP(3) - INTERVAL 2 MINUTE WHERE id=%s", (j.id,))
    post(j.u, f"/assistance/requests/{j.id}/location", {"lat": north(51)[0], "lon": START[1], "accuracy": 8})
    assert float(j.row()["journey_distance_km"]) == pytest.approx(1.0, abs=0.02)


def test_inaccurate_fixes_are_not_counted(j):
    j.request(); j.accept(); j.arrive()
    j.db.query("UPDATE assistance_requests SET journey_last_at = UTC_TIMESTAMP(3) - INTERVAL 5 MINUTE WHERE id=%s", (j.id,))
    post(j.u, f"/assistance/requests/{j.id}/location", {"lat": north(1)[0], "lon": START[1], "accuracy": 900})
    assert float(j.row()["journey_distance_km"]) == 0


def test_straight_line_is_the_minimum_when_updates_are_missing(j):
    j.request(); j.accept(); j.arrive()
    j.set_umusare_position(*north(3.0))                     # no periodic updates reached the server
    j.complete()
    assert float(j.row()["final_distance_km"]) == pytest.approx(haversine_km(*START, *north(3.0)), abs=0.01)


def test_driver_cannot_complete_and_unrelated_umusare_cannot_either(j, app, db):
    j.request(); j.accept(); j.arrive()
    r = post(j.d, f"/assistance/requests/{j.id}/complete")
    assert r.status_code == 403 and j.row()["status"] == "DRIVER_CONNECTED"
    db.user("other@example.com", role="umusare", coop_id=db.cooperative("Other", "OT"))
    o = app.test_client(); login(o, "other@example.com")
    assert post(o, f"/assistance/requests/{j.id}/complete").status_code == 404


def test_journey_cannot_complete_before_arrival(j):
    j.request(); j.accept()
    assert j.complete().status_code == 409 and j.row()["payment_status"] == "ESTIMATED"


def test_tracking_stops_and_exact_journey_points_are_erased(j):
    j.to_payment_pending()
    row = j.row()
    assert row["journey_start_lat"] is None and row["journey_last_lat"] is None and row["pickup_lat"] is None
    assert row["journey_start_approx"] == "-1.94, 30.06"                     # ~1 km precision only
    assert post(j.u, f"/assistance/requests/{j.id}/location", {"lat": START[0], "lon": START[1]}).status_code == 409
    assert post(j.d, f"/assistance/requests/{j.id}/location", {"lat": START[0], "lon": START[1]}).status_code == 409
    assert j.db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (j.driver_id,))[0]["n"] == 0
    details = " ".join(str(r["details"]) for r in j.db.query("SELECT details FROM audit_logs"))
    assert "30.06" not in details and "-1.94" not in details


def test_estimated_fare_during_journey_uses_current_pricing(j):
    j.request(); j.accept(); j.arrive()
    j.drive([1.0, 0.5])
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    assert view["journey"]["distance_so_far_km"] == pytest.approx(1.5, abs=0.02)
    assert view["journey"]["estimated_fare"] == pytest.approx(750, abs=10)
    assert view["journey"]["rate_label"] == "RWF 500 / km" and "payment" not in view


# ---------------------------------------------------------------- payment
def test_full_payment_flow(j):
    j.to_payment_pending()
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    pay = view["payment"]
    assert pay["status"] == "PAYMENT_PENDING" and pay["pay_to_phone"] == "+250788123456"
    assert pay["payee"]["name"] == "Jean Claude M." and pay["payee"]["verified"]
    assert pay["amount_label"].startswith("RWF ") and pay["rate_label"] == "RWF 500 / km"

    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 200
    assert j.row()["payment_status"] == "PAYMENT_SENT" and j.row()["payment_sent_at"] is not None
    umu = j.u.get("/assistance/umusare/status").get_json()
    assert umu["payments"][0]["status"] == "PAYMENT_SENT" and umu["payments"][0]["driver_name"] == "driver"

    assert post(j.u, f"/assistance/requests/{j.id}/payment-received").status_code == 200
    row = j.row()
    assert row["payment_status"] == "PAYMENT_COMPLETED" and row["payment_completed_at"] is not None
    done = j.d.get("/assistance/requests/current").get_json()["request"]
    assert done["status"] == "COMPLETED" and done["payment"]["status"] == "PAYMENT_COMPLETED"
    assert done["summary"]["assistance_id"] == f"AS-{j.id:06d}" and done["summary"]["duration_min"] >= 1
    assert j.u.get("/assistance/umusare/status").get_json()["payments"] == []
    actions = [r["action"] for r in j.db.query("SELECT action FROM audit_logs ORDER BY id")]
    for a in ("JOURNEY_STARTED", "FARE_CALCULATED", "PAYMENT_SENT", "PAYMENT_CONFIRMED"):
        assert a in actions


def test_invalid_payment_transitions(j, app, db, client):
    j.request(); j.accept(); j.arrive()
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 409        # journey not completed
    j.drive([1.0]); j.complete()
    assert post(j.u, f"/assistance/requests/{j.id}/payment-received").status_code == 409    # nothing sent yet
    assert post(j.u, f"/assistance/requests/{j.id}/payment-sent").status_code == 403        # Umusare cannot "send"
    assert post(j.d, f"/assistance/requests/{j.id}/payment-received").status_code == 403    # driver cannot "receive"
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 200
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 409        # already sent
    assert post(client, f"/assistance/requests/{j.id}/payment-received").status_code in (302, 401)
    db.user("other@example.com", role="driver", coop_id=db.cooperative("Other", "OT"))
    o = app.test_client(); login(o, "other@example.com")
    assert post(o, f"/assistance/requests/{j.id}/payment-sent").status_code == 404
    # there is no endpoint that sets a payment status directly
    assert post(j.d, f"/assistance/requests/{j.id}/payment", {"status": "PAYMENT_COMPLETED"}).status_code in (404, 405)
    assert j.row()["payment_status"] == "PAYMENT_SENT"


def test_payment_dispute_and_recovery(j):
    j.to_payment_pending()
    post(j.d, f"/assistance/requests/{j.id}/payment-sent")
    assert post(j.u, f"/assistance/requests/{j.id}/payment-problem", {"note": "Nothing on MoMo"}).status_code == 200
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    assert view["payment"]["status"] == "PAYMENT_DISPUTED" and view["payment"]["dispute_note"] == "Nothing on MoMo"
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 200
    assert post(j.u, f"/assistance/requests/{j.id}/payment-received").status_code == 200
    assert j.row()["payment_status"] == "PAYMENT_COMPLETED"


def test_payment_phone_is_the_registered_number_at_completion(j):
    j.to_payment_pending()
    post(j.u, "/assistance/profile/phone", {"phone": "0788999000"})          # changed later
    post(j.d, f"/assistance/requests/{j.id}/payment-sent", {"phone": "+250700000000", "amount": 1})
    row = j.row()
    assert row["payment_phone"] == "+250788123456" and row["payment_status"] == "PAYMENT_SENT"
    assert j.d.get("/assistance/requests/current").get_json()["request"]["payment"]["pay_to_phone"] == "+250788123456"


def test_rating_and_problem_report(j):
    j.to_payment_pending()
    assert post(j.d, f"/assistance/requests/{j.id}/rate", {"rating": 5}).status_code == 409   # payment not confirmed
    post(j.d, f"/assistance/requests/{j.id}/payment-sent")
    post(j.u, f"/assistance/requests/{j.id}/payment-received")
    assert post(j.d, f"/assistance/requests/{j.id}/rate", {"rating": 9}).status_code == 400
    assert post(j.d, f"/assistance/requests/{j.id}/rate", {"rating": 5, "comment": "Very safe"}).status_code == 200
    assert post(j.d, f"/assistance/requests/{j.id}/rate", {"rating": 4}).status_code == 409   # once only
    assert post(j.d, f"/assistance/requests/{j.id}/report", {"text": "Late arrival"}).status_code == 200
    row = j.row()
    assert (row["rating"], row["problem_report"]) == (5, "Late arrival")


def test_cancelled_request_has_no_payment(j):
    j.request(); j.accept()
    post(j.d, f"/assistance/requests/{j.id}/cancel")
    assert j.row()["payment_status"] == "PAYMENT_CANCELLED"
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 409


def test_request_type_survives_the_whole_journey(j):
    j.to_payment_pending()
    assert j.row()["trigger_source"] == "DRIVER_INITIATED"
    assert j.d.get("/assistance/requests/current").get_json()["request"]["type"] == "DRIVER_INITIATED"


def test_journey_completed_before_fares_existed_shows_no_payment(j):
    j.request(); j.accept(); j.arrive()
    j.db.query("UPDATE assistance_requests SET status='COMPLETED', final_fare=NULL, payment_status='ESTIMATED' WHERE id=%s",
               (j.id,))
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    assert view["status"] == "COMPLETED" and "payment" not in view
    assert post(j.d, f"/assistance/requests/{j.id}/payment-sent").status_code == 409
