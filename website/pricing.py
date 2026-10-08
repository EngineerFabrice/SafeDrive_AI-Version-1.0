# website/pricing.py
"""Admin-configurable fares. The server is the only place a fare is ever calculated.

    fare = base_fee + distance_km x price_per_km, then clamped to [minimum_fare, maximum_fare]
    and rounded to a whole currency unit (half up; RWF has no subunit in practice).

Initial configuration (migration 0004): RWF, 500 per km, base fee 0, minimum 0, no maximum.
A completed journey stores a snapshot of these values, so later changes never alter it.
"""
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from . import audit

FIELDS = ("currency", "price_per_km", "base_fee", "minimum_fare", "maximum_fare")
MAX_AMOUNT = Decimal("1000000")
_CURRENCY = re.compile(r"^[A-Z]{3}$")


class PricingError(ValueError):
    pass


def _dec(value):
    return Decimal(str(value)) if value is not None else None


def get_pricing(cursor):
    cursor.execute("SELECT currency, price_per_km, base_fee, minimum_fare, maximum_fare, updated_at "
                   "FROM pricing_settings WHERE id=1")
    row = cursor.fetchone()
    if row is None:                                  # migration not applied: safe documented default
        return {"currency": "RWF", "price_per_km": Decimal("500"), "base_fee": Decimal("0"),
                "minimum_fare": Decimal("0"), "maximum_fare": None, "updated_at": None}
    return {k: (_dec(row[k]) if k in ("price_per_km", "base_fee", "minimum_fare", "maximum_fare") else row[k])
            for k in row}


def compute_fare(distance_km, pricing):
    """Fare for ``distance_km`` under ``pricing`` (dict with the FIELDS); returns a whole-unit Decimal."""
    km = _dec(round(float(distance_km), 2))
    if km < 0:
        raise PricingError("distance cannot be negative")
    fare = pricing["base_fee"] + km * pricing["price_per_km"]
    fare = max(fare, pricing["minimum_fare"])
    if pricing.get("maximum_fare") is not None:
        fare = min(fare, pricing["maximum_fare"])
    return fare.quantize(Decimal("1"), rounding=ROUND_HALF_UP)


def format_money(amount, currency="RWF"):
    """'RWF 2,300' — amounts are whole units."""
    if amount is None:
        return None
    return f"{currency} {int(Decimal(str(amount)).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}"


def pricing_public(pricing):
    """JSON-safe pricing for display."""
    return {"currency": pricing["currency"], "price_per_km": float(pricing["price_per_km"]),
            "base_fee": float(pricing["base_fee"]), "minimum_fare": float(pricing["minimum_fare"]),
            "maximum_fare": float(pricing["maximum_fare"]) if pricing.get("maximum_fare") is not None else None,
            "rate_label": f"{format_money(pricing['price_per_km'], pricing['currency'])} / km"}


def validate(form):
    """Parse and validate admin input; returns a dict of Decimals (currency as str)."""
    currency = (form.get("currency") or "").strip().upper()
    if not _CURRENCY.match(currency):
        raise PricingError("Currency must be a 3-letter code, e.g. RWF.")
    out = {"currency": currency}
    for key, label in (("price_per_km", "Price per km"), ("base_fee", "Base fee"), ("minimum_fare", "Minimum fare")):
        out[key] = _amount(form.get(key), label, required=True)
    out["maximum_fare"] = _amount(form.get("maximum_fare"), "Maximum fare", required=False)
    if out["price_per_km"] <= 0:
        raise PricingError("Price per km must be greater than 0.")
    if out["maximum_fare"] is not None and out["maximum_fare"] < out["minimum_fare"]:
        raise PricingError("Maximum fare cannot be lower than the minimum fare.")
    return out


def _amount(raw, label, required):
    raw = (raw or "").strip().replace(",", "")
    if raw == "":
        if required:
            raise PricingError(f"{label} is required.")
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        raise PricingError(f"{label} must be a number.") from None
    if not value.is_finite() or value < 0 or value > MAX_AMOUNT:
        raise PricingError(f"{label} must be between 0 and {MAX_AMOUNT:,}.")
    return value.quantize(Decimal("0.01"))


def update_pricing(cursor, admin_id, new):
    previous = get_pricing(cursor)
    cursor.execute("INSERT INTO pricing_settings (id, currency, price_per_km, base_fee, minimum_fare, maximum_fare, "
                   "updated_by) VALUES (1, %s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE currency=VALUES(currency), "
                   "price_per_km=VALUES(price_per_km), base_fee=VALUES(base_fee), minimum_fare=VALUES(minimum_fare), "
                   "maximum_fare=VALUES(maximum_fare), updated_by=VALUES(updated_by)",
                   (new["currency"], new["price_per_km"], new["base_fee"], new["minimum_fare"], new["maximum_fare"],
                    admin_id))
    as_text = lambda p: {k: (str(p[k]) if p[k] is not None else None) for k in FIELDS}   # noqa: E731
    audit.record(audit.PRICING_UPDATED, actor_id=admin_id, target_type="pricing_settings", target_id=1,
                 details={"previous": as_text(previous), "new": as_text(new)}, cursor=cursor)
