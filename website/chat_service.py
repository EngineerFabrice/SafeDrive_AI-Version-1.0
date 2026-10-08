# website/chat_service.py
"""Secure internal chat: role-aware and cooperative-aware, authorized on every call.

Who may talk to whom (decided here from the database, never from the browser):

* Admin   - anyone (global support / security conversations).
* Manager - Admin, and drivers / Umusare / managers of their OWN cooperative.
* Driver / Umusare - their own cooperative's manager (always, also while PENDING or
  REJECTED, so they can sort out verification) and, once VERIFIED, other VERIFIED
  drivers / Umusare of the same cooperative.
* A conversation an Admin is part of stays open for the other participant to reply.
* Nobody can reach another cooperative's members (except Admin).

Every operation re-checks the rule: conversation creation, listing, reading, sending
and contact search. Conversations are addressed by an unguessable public id; people
are addressed by a signed, per-viewer contact token. Unknown, foreign and forbidden
targets all answer the same 404, so the API cannot be used to enumerate users.
Message text is never written to the audit log or to notifications.
"""
import re
import secrets

import pymysql
from flask import current_app
from itsdangerous import BadSignature, URLSafeSerializer

from . import ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE
from . import audit
from . import notifications as notes
from .assistance_service import AssistanceError, transaction, utcnow
from .cooperative_service import cooperative_manager

BODY_MAX = 2000
PAGE = 100
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9_-]{22}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
NOT_AVAILABLE = AssistanceError(404, "NOT_FOUND", "This conversation or contact is not available.")
ROLE_LABELS = {ROLE_ADMIN: "Admin", ROLE_MANAGER: "Manager", ROLE_DRIVER: "Driver", ROLE_UMUSARE: "Umusare"}
VERIFICATION_GREETING = "Hello, I have registered and I am waiting for cooperative verification."

_PARTY_SQL = (
    "SELECT u.id, u.username, u.email, u.role, u.is_active, m.cooperative_id, m.status AS membership_status, "
    "m.member_role, c.name AS cooperative, "
    "COALESCE(IF(u.role='driver', dp.verification_status, up.verification_status), 'PENDING') AS verification "
    "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
    "LEFT JOIN cooperatives c ON c.id = m.cooperative_id "
    "LEFT JOIN driver_profiles dp ON dp.user_id = u.id LEFT JOIN umusare_profiles up ON up.user_id = u.id ")


# ---------------------------------------------------------------- policy
def _party(cursor, user_id):
    cursor.execute(_PARTY_SQL + "WHERE u.id=%s", (user_id,))
    return cursor.fetchone()


def _parties(cursor, ids):
    ids = sorted({int(i) for i in ids})
    if not ids:
        return {}
    cursor.execute(_PARTY_SQL + f"WHERE u.id IN ({','.join(['%s'] * len(ids))})", ids)
    return {r["id"]: r for r in cursor.fetchall()}


def _cooperative_of(p):
    """The cooperative a person may chat within, or None."""
    if p["role"] not in (ROLE_DRIVER, ROLE_UMUSARE, ROLE_MANAGER) or not p["cooperative_id"]:
        return None
    if p["member_role"] != p["role"] or p["membership_status"] == "REVOKED":
        return None
    if p["role"] == ROLE_MANAGER and p["membership_status"] != "APPROVED":
        return None
    return p["cooperative_id"]


def _may_initiate(a, b):
    """Directional rule: may ``a`` start a conversation with ``b``? (activity checked separately)"""
    if a["id"] == b["id"]:
        return False
    if a["role"] == ROLE_ADMIN:
        return True
    if b["role"] == ROLE_ADMIN:
        return a["role"] == ROLE_MANAGER
    coop = _cooperative_of(a)
    if coop is None or _cooperative_of(b) != coop:
        return False
    if ROLE_MANAGER in (a["role"], b["role"]):
        return True
    return a["verification"] == "VERIFIED" and b["verification"] == "VERIFIED"


def may_converse(a, b):
    """May ``a`` and ``b`` have (or keep using) a conversation? Symmetric, so Admin-started ones allow replies."""
    if not (a and b and a["is_active"]):
        return False
    return _may_initiate(a, b) or _may_initiate(b, a)


def may_start(a, b):
    return bool(a and b and a["is_active"] and b["is_active"] and _may_initiate(a, b))


# ---------------------------------------------------------------- contact tokens (signed, per viewer)
def _serializer():
    return URLSafeSerializer(current_app.config["SECRET_KEY"], salt="safedrive-chat-contact")


def contact_token(viewer_id, user_id):
    return _serializer().dumps([int(viewer_id), int(user_id)])


def _token_target(viewer_id, token):
    try:
        viewer, target = _serializer().loads(str(token or ""))
    except (BadSignature, ValueError, TypeError):
        raise NOT_AVAILABLE
    if int(viewer) != int(viewer_id):
        raise NOT_AVAILABLE
    return int(target)


# ---------------------------------------------------------------- helpers
def _iso(dt):
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


def _person(p, viewer_role):
    out = {"name": p["username"], "role": p["role"], "role_label": ROLE_LABELS.get(p["role"], p["role"]),
           "cooperative": p["cooperative"] if p["role"] != ROLE_ADMIN and _cooperative_of(p) else None,
           "verified": p["verification"] == "VERIFIED" if p["role"] in (ROLE_DRIVER, ROLE_UMUSARE) else None,
           "active": bool(p["is_active"])}
    if viewer_role == ROLE_ADMIN:
        out["email"] = p["email"]
    return out


def _conversation_type(a, b):
    if ROLE_ADMIN in (a["role"], b["role"]):
        return "ADMINISTRATIVE"
    if ROLE_MANAGER in (a["role"], b["role"]):
        member = b if a["role"] == ROLE_MANAGER else a
        if member["role"] in (ROLE_DRIVER, ROLE_UMUSARE) and member["verification"] != "VERIFIED":
            return "VERIFICATION"
        return "SUPPORT"
    return "DIRECT"


def clean_body(raw):
    body = _CONTROL.sub("", str(raw or "")).replace("\r\n", "\n").strip()
    if not body:
        raise AssistanceError(400, "EMPTY", "Type a message first.")
    if len(body) > BODY_MAX:
        raise AssistanceError(400, "TOO_LONG", f"Messages are limited to {BODY_MAX} characters.")
    return body


def _load_conversation(cursor, actor_id, public_id, lock=False):
    """(conversation, me, other) if the actor is a participant AND the pair is still allowed; else 404."""
    if not _PUBLIC_ID.match(str(public_id or "")):
        raise NOT_AVAILABLE
    cursor.execute("SELECT c.id, c.public_id, c.conversation_type, c.cooperative_id, c.created_by, c.created_at "
                   "FROM conversations c JOIN conversation_participants p ON p.conversation_id = c.id AND p.user_id=%s "
                   "WHERE c.public_id=%s" + (" FOR UPDATE" if lock else ""), (actor_id, public_id))
    conv = cursor.fetchone()
    if conv is None:
        raise NOT_AVAILABLE
    cursor.execute("SELECT user_id FROM conversation_participants WHERE conversation_id=%s AND user_id<>%s",
                   (conv["id"], actor_id))
    other_row = cursor.fetchone()
    parties = _parties(cursor, [actor_id] + ([other_row["user_id"]] if other_row else []))
    me, other = parties.get(int(actor_id)), parties.get(other_row["user_id"]) if other_row else None
    if not may_converse(me, other):
        raise NOT_AVAILABLE
    return conv, me, other


# ---------------------------------------------------------------- conversations
def _get_or_create(cursor, me, other):
    """Direct conversation between two allowed people (one per pair, race-safe via the unique key)."""
    key = f"{min(me['id'], other['id'])}:{max(me['id'], other['id'])}"
    cursor.execute("SELECT public_id FROM conversations WHERE direct_key=%s", (key,))
    row = cursor.fetchone()
    if row:
        return row["public_id"], False
    now = utcnow()
    shared = _cooperative_of(me) if _cooperative_of(me) == _cooperative_of(other) else None
    public_id = secrets.token_urlsafe(16)[:22]
    cursor.execute("INSERT INTO conversations (public_id, conversation_type, direct_key, cooperative_id, created_by, "
                   "created_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                   (public_id, _conversation_type(me, other), key, shared, me["id"], now, now))
    conv_id = cursor.lastrowid
    cursor.execute("INSERT INTO conversation_participants (conversation_id, user_id, joined_at, last_read_at) "
                   "VALUES (%s,%s,%s,%s), (%s,%s,%s,NULL)", (conv_id, me["id"], now, now, conv_id, other["id"], now))
    audit.record(audit.CHAT_CONVERSATION_CREATED, actor_id=me["id"], target_type="conversation", target_id=public_id,
                 cooperative_id=shared, details={"with_role": other["role"]}, cursor=cursor)
    if other["role"] == ROLE_MANAGER and me["role"] != ROLE_ADMIN:
        audit.record(audit.MANAGER_CONTACTED, actor_id=me["id"], target_type="user", target_id=other["id"],
                     cooperative_id=shared, cursor=cursor)
    elif other["role"] == ROLE_ADMIN:
        audit.record(audit.ADMIN_CONTACTED, actor_id=me["id"], target_type="user", target_id=other["id"],
                     cursor=cursor)
    return public_id, True


def _open_with(actor_id, target_id):
    for attempt in (1, 2):
        try:
            with transaction() as cursor:
                parties = _parties(cursor, [actor_id, target_id])
                me, other = parties.get(int(actor_id)), parties.get(int(target_id))
                if not may_start(me, other):
                    # an existing conversation (e.g. started by Admin) may still be continued
                    key = f"{min(int(actor_id), int(target_id))}:{max(int(actor_id), int(target_id))}"
                    cursor.execute("SELECT public_id FROM conversations WHERE direct_key=%s", (key,))
                    row = cursor.fetchone()
                    if row and may_converse(me, other) and other["is_active"]:
                        return {"conversation": row["public_id"], "created": False}
                    raise NOT_AVAILABLE
                public_id, created = _get_or_create(cursor, me, other)
                return {"conversation": public_id, "created": created}
        except pymysql.err.IntegrityError:
            if attempt == 2:                     # concurrent create of the same pair: the other one won
                raise


def open_conversation(actor_id, token):
    """Start (or reopen) a conversation with a contact token from contacts()."""
    return _open_with(actor_id, _token_target(actor_id, token))


def contact_manager(actor_id):
    """Driver / Umusare: open the conversation with their own cooperative's manager."""
    with transaction() as cursor:
        me = _party(cursor, actor_id)
        if not me or me["role"] not in (ROLE_DRIVER, ROLE_UMUSARE):
            raise AssistanceError(403, "FORBIDDEN", "Only drivers and Umusare use this shortcut.")
        coop = _cooperative_of(me)
        manager = cooperative_manager(cursor, coop) if coop else None
    if manager is None:
        raise AssistanceError(404, "NO_MANAGER", "Your cooperative has no active manager yet. "
                                                 "Please contact SafeDrive support.")
    result = _open_with(actor_id, manager["user_id"])
    result["greeting"] = VERIFICATION_GREETING if me["verification"] != "VERIFIED" else ""
    return result


def list_conversations(actor_id):
    with transaction() as cursor:
        cursor.execute(
            "SELECT c.public_id, c.conversation_type, c.created_at, c.last_message_at, o.user_id AS other_id, "
            "(SELECT COUNT(*) FROM messages mm WHERE mm.conversation_id = c.id AND mm.deleted_at IS NULL "
            " AND (mm.sender_id IS NULL OR mm.sender_id <> p.user_id) "
            " AND (p.last_read_at IS NULL OR mm.created_at > p.last_read_at)) AS unread, "
            "(SELECT mm.body FROM messages mm WHERE mm.conversation_id = c.id AND mm.deleted_at IS NULL "
            " ORDER BY mm.id DESC LIMIT 1) AS last_body, "
            "(SELECT mm.sender_id FROM messages mm WHERE mm.conversation_id = c.id AND mm.deleted_at IS NULL "
            " ORDER BY mm.id DESC LIMIT 1) AS last_sender "
            "FROM conversation_participants p JOIN conversations c ON c.id = p.conversation_id "
            "JOIN conversation_participants o ON o.conversation_id = c.id AND o.user_id <> p.user_id "
            "WHERE p.user_id=%s ORDER BY COALESCE(c.last_message_at, c.created_at) DESC LIMIT 100", (actor_id,))
        rows = cursor.fetchall()
        parties = _parties(cursor, [actor_id] + [r["other_id"] for r in rows])
    me = parties.get(int(actor_id))
    out = []
    for r in rows:
        other = parties.get(r["other_id"])
        if not may_converse(me, other):
            continue                                   # e.g. a member who has left the cooperative
        preview = (r["last_body"] or "").replace("\n", " ")
        out.append({"id": r["public_id"], "type": r["conversation_type"], "with": _person(other, me["role"]),
                    "last_message": (("You: " if r["last_sender"] == me["id"] else "") + preview)[:90] or None,
                    "last_at": _iso(r["last_message_at"] or r["created_at"]), "unread": int(r["unread"] or 0)})
    return out


def unread_total(actor_id):
    with transaction() as cursor:
        nots = notes.unread_count(cursor, actor_id)
    return {"messages": sum(c["unread"] for c in list_conversations(actor_id)), "notifications": nots}


def read_messages(actor_id, public_id, after=0):
    """Messages of one conversation (all, or newer than ``after``); marks them read."""
    try:
        after = max(0, int(after or 0))
    except (TypeError, ValueError):
        after = 0
    with transaction() as cursor:
        conv, me, other = _load_conversation(cursor, actor_id, public_id)
        if after:
            cursor.execute("SELECT id, sender_id, body, created_at FROM messages WHERE conversation_id=%s AND id>%s "
                           "AND deleted_at IS NULL ORDER BY id LIMIT %s", (conv["id"], after, PAGE))
            rows = cursor.fetchall()
        else:
            cursor.execute("SELECT id, sender_id, body, created_at FROM messages WHERE conversation_id=%s "
                           "AND deleted_at IS NULL ORDER BY id DESC LIMIT %s", (conv["id"], PAGE))
            rows = list(reversed(cursor.fetchall()))
        cursor.execute("UPDATE conversation_participants SET last_read_at=%s WHERE conversation_id=%s AND user_id=%s",
                       (utcnow(), conv["id"], actor_id))
    names = {me["id"]: me["username"], other["id"]: other["username"]}
    return {"conversation": {"id": conv["public_id"], "type": conv["conversation_type"],
                             "with": _person(other, me["role"]), "can_send": bool(other["is_active"])},
            "messages": [{"id": r["id"], "mine": r["sender_id"] == me["id"], "sender": names.get(r["sender_id"], "—"),
                          "body": r["body"], "at": _iso(r["created_at"])} for r in rows]}


def send_message(actor_id, public_id, raw_body):
    body = clean_body(raw_body)
    with transaction() as cursor:
        conv, me, other = _load_conversation(cursor, actor_id, public_id, lock=True)
        if not other["is_active"]:
            raise AssistanceError(409, "INACTIVE", "This account is deactivated and cannot receive messages.")
        now = utcnow()
        cursor.execute("INSERT INTO messages (conversation_id, sender_id, body, created_at) VALUES (%s,%s,%s,%s)",
                       (conv["id"], me["id"], body, now))
        message_id = cursor.lastrowid
        cursor.execute("UPDATE conversations SET last_message_at=%s, updated_at=%s WHERE id=%s", (now, now, conv["id"]))
        cursor.execute("UPDATE conversation_participants SET last_read_at=%s WHERE conversation_id=%s AND user_id=%s",
                       (now, conv["id"], me["id"]))
        audit.record(audit.CHAT_MESSAGE_SENT, actor_id=me["id"], target_type="conversation",
                     target_id=conv["public_id"], cooperative_id=conv["cooperative_id"], cursor=cursor)
        if (other["role"] == ROLE_MANAGER and me["role"] in (ROLE_DRIVER, ROLE_UMUSARE)
                and me["verification"] != "VERIFIED"):
            link = f"/chat/?c={conv['public_id']}"
            if not notes.has_unread(cursor, other["id"], notes.SUPPORT_MESSAGE, link):
                notes.notify(cursor, other["id"], notes.SUPPORT_MESSAGE,
                             f"New verification/support message from {me['username']}", link)
    return {"id": message_id, "mine": True, "sender": me["username"], "body": body, "at": _iso(now)}


def contacts(actor_id, query=""):
    """People the actor may start a conversation with (server-filtered; max 25)."""
    q = (query or "").strip()[:60]
    like = f"%{q}%"
    with transaction() as cursor:
        me = _party(cursor, actor_id)
        if not me or not me["is_active"]:
            raise AssistanceError(403, "FORBIDDEN", "You do not have permission for this action.")
        where, args = ["u.is_active = 1", "u.id <> %s"], [actor_id]
        if q:
            where.append("(u.username LIKE %s" + (" OR u.email LIKE %s)" if me["role"] == ROLE_ADMIN else ")"))
            args += [like, like] if me["role"] == ROLE_ADMIN else [like]
        coop = _cooperative_of(me)
        if me["role"] == ROLE_ADMIN:
            pass
        elif me["role"] == ROLE_MANAGER:
            where.append("(u.role = 'admin' OR m.cooperative_id = %s)" if coop else "u.role = 'admin'")
            args += [coop] if coop else []
        elif coop:
            where.append("m.cooperative_id = %s")
            args.append(coop)
        else:
            return []
        cursor.execute(_PARTY_SQL + "WHERE " + " AND ".join(where)
                       + " ORDER BY FIELD(u.role,'manager','admin','umusare','driver'), u.username LIMIT 60", args)
        rows = cursor.fetchall()
    out = []
    for p in rows:
        if may_start(me, p):                           # the policy itself is the final filter
            out.append({**_person(p, me["role"]), "token": contact_token(actor_id, p["id"])})
        if len(out) >= 25:
            break
    return out
