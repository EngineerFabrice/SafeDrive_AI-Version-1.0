# website/notifications.py
"""Lightweight in-app notifications (table `notifications`).

Used for verification decisions and new verification requests. Titles are short,
built by the server, and never contain chat message text or locations.
"""
from .assistance_service import transaction, utcnow

VERIFICATION_REQUESTED = "VERIFICATION_REQUESTED"
VERIFICATION_APPROVED = "VERIFICATION_APPROVED"
VERIFICATION_REJECTED = "VERIFICATION_REJECTED"
VERIFICATION_INFO_REQUESTED = "VERIFICATION_INFO_REQUESTED"
VERIFICATION_SUSPENDED = "VERIFICATION_SUSPENDED"
SUPPORT_MESSAGE = "SUPPORT_MESSAGE"
MANAGER_ASSIGNED = "MANAGER_ASSIGNED"


def notify(cursor, user_id, kind, title, link=None):
    """Insert one notification inside the caller's transaction."""
    cursor.execute("INSERT INTO notifications (user_id, kind, title, link, created_at) VALUES (%s,%s,%s,%s,%s)",
                   (user_id, kind, title[:200], link, utcnow()))


def has_unread(cursor, user_id, kind, link):
    cursor.execute("SELECT 1 FROM notifications WHERE user_id=%s AND kind=%s AND link<=>%s AND read_at IS NULL LIMIT 1",
                   (user_id, kind, link))
    return cursor.fetchone() is not None


def unread_count(cursor, user_id):
    cursor.execute("SELECT COUNT(*) AS n FROM notifications WHERE user_id=%s AND read_at IS NULL", (user_id,))
    return cursor.fetchone()["n"]


def recent(user_id, limit=20):
    """The user's own notifications only (newest first)."""
    with transaction() as cursor:
        cursor.execute("SELECT id, kind, title, link, created_at, read_at FROM notifications WHERE user_id=%s "
                       "ORDER BY created_at DESC, id DESC LIMIT %s", (user_id, limit))
        rows = cursor.fetchall()
    return [{"id": r["id"], "kind": r["kind"], "title": r["title"], "link": r["link"],
             "created_at": r["created_at"].isoformat(timespec="seconds") + "Z", "unread": r["read_at"] is None}
            for r in rows]


def mark_all_read(user_id):
    with transaction() as cursor:
        cursor.execute("UPDATE notifications SET read_at=%s WHERE user_id=%s AND read_at IS NULL", (utcnow(), user_id))
