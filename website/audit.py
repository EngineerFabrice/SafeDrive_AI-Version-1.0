# website/audit.py
"""Append-only audit log (table `audit_logs`).

`record()` never raises: an audit write failure is logged and must not break
the user's action. Details are short JSON and must never contain passwords,
face images, feature vectors or precise locations.
"""
import json
import logging
from datetime import datetime, timezone

from flask import has_request_context, request

from . import get_connection

log = logging.getLogger(__name__)

# Action names used so far; later phases add assessment and assistance actions.
LOGIN_SUCCEEDED = "LOGIN_SUCCEEDED"
LOGIN_FAILED = "LOGIN_FAILED"
LOGOUT = "LOGOUT"
USER_REGISTERED = "USER_REGISTERED"
ROLE_CHANGED = "ROLE_CHANGED"
MEMBERSHIP_CHANGED = "MEMBERSHIP_CHANGED"
USER_DELETED = "USER_DELETED"
COOPERATIVE_CREATED = "COOPERATIVE_CREATED"
MONITORING_STARTED = "MONITORING_STARTED"
MONITORING_STOPPED = "MONITORING_STOPPED"
DEV_SEED = "DEV_SEED"                      # development/demo account or cooperative created by website.dev_seed
# Assistance workflow (details never contain coordinates)
ASSISTANCE_REQUESTED = "ASSISTANCE_REQUESTED"
ASSISTANCE_MATCHING = "ASSISTANCE_MATCHING"          # a matching round offered the request to Umusare
ASSISTANCE_NO_UMUSARE = "ASSISTANCE_NO_UMUSARE_AVAILABLE"
ASSISTANCE_DECLINED = "ASSISTANCE_DECLINED"
ASSISTANCE_ACCEPTED = "ASSISTANCE_ACCEPTED"
ASSISTANCE_CONNECTED = "ASSISTANCE_CONNECTED"
ASSISTANCE_CANCELLED = "ASSISTANCE_CANCELLED"
ASSISTANCE_COMPLETED = "ASSISTANCE_COMPLETED"
UMUSARE_AVAILABILITY_CHANGED = "UMUSARE_AVAILABILITY_CHANGED"
# Journey, fare and payment (no coordinates; amounts and distances only)
JOURNEY_STARTED = "JOURNEY_STARTED"
FARE_CALCULATED = "FARE_CALCULATED"
PAYMENT_SENT = "PAYMENT_SENT"
PAYMENT_CONFIRMED = "PAYMENT_CONFIRMED"
PAYMENT_DISPUTED = "PAYMENT_DISPUTED"
ASSISTANCE_RATING = "RATING_SUBMITTED"
PROBLEM_REPORTED = "PROBLEM_REPORTED"
# Administration
PRICING_UPDATED = "PRICE_UPDATED"
ACCOUNT_STATUS_CHANGED = "ACCOUNT_STATUS_CHANGED"
UMUSARE_VERIFICATION_CHANGED = "UMUSARE_VERIFICATION_CHANGED"
PHONE_UPDATED = "PHONE_UPDATED"
REGISTRATION_DUPLICATE = "REGISTRATION_DUPLICATE"        # registration with an existing email (no user revealed)
# Cooperative management and member verification
COOPERATIVE_UPDATED = "COOPERATIVE_UPDATED"
MANAGER_ASSIGNED = "MANAGER_ASSIGNED"
MANAGER_CHANGED = "MANAGER_CHANGED"
USER_VERIFICATION_REQUESTED = "USER_VERIFICATION_REQUESTED"
USER_VERIFIED = "USER_VERIFIED"
USER_VERIFICATION_REJECTED = "USER_VERIFICATION_REJECTED"
USER_VERIFICATION_INFO_REQUESTED = "USER_VERIFICATION_INFO_REQUESTED"
USER_SUSPENDED = "USER_SUSPENDED"
# Internal chat (never the message text)
CHAT_CONVERSATION_CREATED = "CHAT_CONVERSATION_CREATED"
CHAT_MESSAGE_SENT = "CHAT_MESSAGE_SENT"
MANAGER_CONTACTED = "MANAGER_CONTACTED"
ADMIN_CONTACTED = "ADMIN_CONTACTED"
# Email verification, legal acceptance, groups, vehicles, nearby visibility
EMAIL_OTP_SENT = "EMAIL_OTP_SENT"
EMAIL_OTP_FAILED = "EMAIL_OTP_FAILED"
EMAIL_DELIVERY_FAILED = "EMAIL_DELIVERY_FAILED"
EMAIL_VERIFIED = "EMAIL_VERIFIED"
TERMS_ACCEPTED = "TERMS_ACCEPTED"
GROUP_CREATED = "GROUP_CREATED"
GROUP_UPDATED = "GROUP_UPDATED"
GROUP_MEMBER_ADDED = "GROUP_MEMBER_ADDED"
GROUP_MEMBER_REMOVED = "GROUP_MEMBER_REMOVED"
VEHICLE_UPDATED = "VEHICLE_UPDATED"
NEARBY_VISIBILITY_CHANGED = "NEARBY_VISIBILITY_CHANGED"


def record(action, actor_id=None, target_type=None, target_id=None, cooperative_id=None,
           details=None, cursor=None):
    """Insert one audit row. Pass ``cursor`` to write inside the caller's transaction."""
    row = (datetime.now(timezone.utc).replace(tzinfo=None), actor_id, action, target_type,
           None if target_id is None else str(target_id), cooperative_id,
           request.remote_addr if has_request_context() else None,
           json.dumps(details, separators=(",", ":"))[:1000] if details else None)
    sql = ("INSERT INTO audit_logs (occurred_at, actor_user_id, action, target_type, target_id, "
           "cooperative_id, ip_address, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)")
    if cursor is not None:
        cursor.execute(sql, row)   # caller's transaction: errors propagate and roll back together
        return True
    try:
        conn = get_connection(connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, row)
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as exc:
        log.warning("Audit record %s not saved: %s", action, type(exc).__name__)
        return False
