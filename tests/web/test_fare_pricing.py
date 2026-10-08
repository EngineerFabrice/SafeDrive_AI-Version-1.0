"""Fare formula, configurable pricing, rounding, formatting, snapshots and the admin pricing page."""
from decimal import Decimal

import pytest

from tests.web.conftest import login
from tests.web.journey_helpers import Journey
from website.pricing import PricingError, compute_fare, format_money, validate


def pricing(per_km="500", base="0", minimum="0", maximum=None, currency="RWF"):
    return {"currency": currency, "price_per_km": Decimal(per_km), "base_fee": Decimal(base),
            "minimum_fare": Decimal(minimum), "maximum_fare": Decimal(maximum) if maximum else None}


# ---------------------------------------------------------------- pure formula
def test_fare_is_distance_times_rate():
    assert compute_fare(4.6, pricing()) == Decimal("2300")


def test_configurable_rate_base_fee_minimum_and_maximum():
    assert compute_fare(4.6, pricing(per_km="600")) == Decimal("2760")
    assert compute_fare(4.6, pricing(base="200")) == Decimal("2500")
    assert compute_fare(0.5, pricing(minimum="1000")) == Decimal("1000")        # minimum applies
    assert compute_fare(40, pricing(maximum="15000")) == Decimal("15000")       # maximum applies
    assert compute_fare(0, pricing()) == Decimal("0")


def test_rounding_to_whole_currency_half_up():
    assert compute_fare(1.03, pricing()) == Decimal("515")
    assert compute_fare(1.01, pricing(per_km="333")) == Decimal("336")          # 336.33 -> 336
    assert compute_fare(0.01, pricing(per_km="50")) == Decimal("1")             # 0.50 -> 1 (half up)
    assert compute_fare(2.004, pricing()) == Decimal("1000")                    # distance kept to 0.01 km


def test_negative_distance_rejected():
    with pytest.raises(PricingError):
        compute_fare(-1, pricing())


def test_rwf_formatting():
    assert format_money(2300) == "RWF 2,300"
    assert format_money(Decimal("1234567.4"), "RWF") == "RWF 1,234,567"
    assert format_money(None) is None


@pytest.mark.parametrize("form, message", [
    ({"currency": "rw", "price_per_km": "500", "base_fee": "0", "minimum_fare": "0"}, "Currency"),
    ({"currency": "RWF", "price_per_km": "abc", "base_fee": "0", "minimum_fare": "0"}, "number"),
    ({"currency": "RWF", "price_per_km": "-5", "base_fee": "0", "minimum_fare": "0"}, "between"),
    ({"currency": "RWF", "price_per_km": "0", "base_fee": "0", "minimum_fare": "0"}, "greater than 0"),
    ({"currency": "RWF", "price_per_km": "500", "base_fee": "0", "minimum_fare": "900", "maximum_fare": "800"},
     "Maximum"),
])
def test_pricing_validation(form, message):
    with pytest.raises(PricingError) as err:
        validate(form)
    assert message in str(err.value)


# ---------------------------------------------------------------- database: default, admin page, snapshot

@pytest.mark.db
def test_initial_configuration_is_500_rwf_per_km(db):
    r = db.query("SELECT currency, price_per_km, base_fee, minimum_fare, maximum_fare FROM pricing_settings WHERE id=1")[0]
    assert (r["currency"], r["price_per_km"], r["base_fee"], r["minimum_fare"], r["maximum_fare"]) == \
        ("RWF", Decimal("500.00"), Decimal("0.00"), Decimal("0.00"), None)


@pytest.mark.db
def test_admin_updates_pricing_with_audit(client, db):
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    assert "500" in client.get("/admin/pricing").get_data(as_text=True)
    r = client.post("/admin/pricing", data={"currency": "RWF", "price_per_km": "600", "base_fee": "100",
                                            "minimum_fare": "500", "maximum_fare": ""})
    assert r.status_code == 302
    row = db.query("SELECT price_per_km, base_fee, minimum_fare, maximum_fare FROM pricing_settings WHERE id=1")[0]
    assert (row["price_per_km"], row["base_fee"], row["minimum_fare"], row["maximum_fare"]) == \
        (Decimal("600.00"), Decimal("100.00"), Decimal("500.00"), None)
    audit = db.query("SELECT details FROM audit_logs WHERE action='PRICE_UPDATED'")[0]["details"]
    assert '"price_per_km":"500' in audit and '"price_per_km":"600' in audit


@pytest.mark.db
def test_invalid_pricing_rejected_and_unchanged(client, db):
    db.query("INSERT IGNORE INTO pricing_settings (id) VALUES (1)")
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    client.post("/admin/pricing", data={"currency": "RWF", "price_per_km": "-1", "base_fee": "0", "minimum_fare": "0"})
    assert db.query("SELECT price_per_km FROM pricing_settings")[0]["price_per_km"] == Decimal("500.00")


@pytest.mark.db
@pytest.mark.parametrize("role", ["driver", "umusare", "manager"])
def test_non_admin_cannot_change_pricing(client, db, role):
    db.query("INSERT IGNORE INTO pricing_settings (id) VALUES (1)")
    db.user("x@example.com", role=role, coop_id=db.cooperative())
    login(client, "x@example.com")
    r = client.post("/admin/pricing", data={"currency": "RWF", "price_per_km": "1", "base_fee": "0", "minimum_fare": "0"},
                    headers={"Accept": "application/json"})
    assert r.status_code == 403
    assert db.query("SELECT price_per_km FROM pricing_settings")[0]["price_per_km"] == Decimal("500.00")


@pytest.mark.db
def test_completed_journey_keeps_its_price_after_rate_change(app, db, client):
    db.query("INSERT IGNORE INTO pricing_settings (id) VALUES (1)")
    j = Journey(app, db)
    j.to_payment_pending()
    before = j.row()
    assert before["fare_price_per_km"] == Decimal("500.00")
    db.user("admin@example.com", role="admin")
    login(client, "admin@example.com")
    client.post("/admin/pricing", data={"currency": "RWF", "price_per_km": "600", "base_fee": "0", "minimum_fare": "0"})
    after = j.row()
    assert (after["final_fare"], after["fare_price_per_km"], after["final_distance_km"]) == \
        (before["final_fare"], Decimal("500.00"), before["final_distance_km"])
    view = j.d.get("/assistance/requests/current").get_json()["request"]
    assert view["payment"]["rate_label"] == "RWF 500 / km"
