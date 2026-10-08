# website/geo.py
"""Geographic helpers and Umusare ranking for the assistance workflow.

Pure functions (no database, no map provider) so matching works and is tested
locally. Google Maps is only used by the browser for display/navigation.
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence

EARTH_RADIUS_KM = 6371.0088
APPROX_DECIMALS = 2          # ~1.1 km grid: the only driver location shown before acceptance
SAME_COOPERATIVE_BONUS_KM = 0.5   # same cooperative may only beat a candidate that is < 0.5 km closer
RELIABILITY_PENALTY_KM = 1.0      # at most +1 km for a candidate who declines most requests
MIN_REQUESTS_FOR_RELIABILITY = 3


def valid_coordinates(lat, lon) -> bool:
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180 \
        and not (lat == 0 and lon == 0)          # (0, 0) is almost always a failed GPS fix


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp, dl = p2 - p1, math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def approximate(lat, lon, decimals=APPROX_DECIMALS):
    """Coarse grid point (~1.1 km) that reveals the area but not the exact position."""
    return round(float(lat), decimals), round(float(lon), decimals)


def approximate_distance_label(km: float) -> str:
    """Distance shown before acceptance: rounded to 0.5 km so it cannot pinpoint the driver."""
    if km < 1.0:
        return "less than 1 km"
    return f"about {math.ceil(km * 2) / 2:.1f} km"


@dataclass(frozen=True)
class Candidate:
    umusare_id: int
    lat: float
    lon: float
    cooperative_id: int
    requests_received: int = 0
    requests_accepted: int = 0


@dataclass(frozen=True)
class RankedCandidate:
    candidate: Candidate
    distance_km: float
    same_cooperative: bool
    score_km: float                 # lower is better: distance adjusted by bounded bonus / penalty


def rank_candidates(lat: float, lon: float, driver_cooperative_id: Optional[int],
                    candidates: Sequence[Candidate], radius_km: float):
    """Eligible candidates within ``radius_km``, best first.

    Distance dominates. Being in the driver's cooperative subtracts at most
    SAME_COOPERATIVE_BONUS_KM, so it only breaks near-ties; a much closer
    Umusare from another cooperative always ranks first. A poor acceptance
    record adds at most RELIABILITY_PENALTY_KM.
    """
    ranked = []
    for c in candidates:
        d = haversine_km(lat, lon, c.lat, c.lon)
        if d > radius_km:
            continue
        same = driver_cooperative_id is not None and c.cooperative_id == driver_cooperative_id
        score = d - (SAME_COOPERATIVE_BONUS_KM if same else 0.0)
        if c.requests_received >= MIN_REQUESTS_FOR_RELIABILITY:
            decline_rate = 1.0 - min(1.0, c.requests_accepted / c.requests_received)
            score += RELIABILITY_PENALTY_KM * decline_rate
        ranked.append(RankedCandidate(c, d, same, score))
    ranked.sort(key=lambda r: (r.score_km, r.distance_km, r.candidate.umusare_id))
    return ranked
