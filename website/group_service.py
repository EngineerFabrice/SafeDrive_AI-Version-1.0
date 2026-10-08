# website/group_service.py
"""Groups inside a cooperative (Admin -> Cooperative -> Manager -> Groups -> Drivers / Umusare).

A group belongs to exactly one cooperative. A manager manages only the groups of their own
cooperative (others answer 404); admin manages all. A member can join only a group of their
own cooperative; the database enforces this too (composite foreign key on the membership).
Group ids and cooperative ids sent by the browser are only *requests*: scope always comes from
the authenticated user's database relationships.
"""
import re

from . import ROLE_ADMIN, ROLE_MANAGER
from . import audit
from .assistance_service import AssistanceError, transaction, utcnow
from .cooperative_service import actor_role, manager_cooperative, member_code

_NAME = re.compile(r"^[\w][\w .,'&()/-]{1,78}[\w.)]$", re.UNICODE)
NOT_FOUND = "Group not found."


def group_code(group_id):
    return f"GRP-{int(group_id):04d}"


def clean_name(raw):
    name = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not _NAME.match(name):
        raise AssistanceError(400, "INVALID_NAME", "Group names are 3-80 characters (letters, numbers, spaces, - . , ' & ( ) /).")
    return name


def _scope(cursor, actor_id):
    """(role, cooperative id the actor is limited to, or None for admin)."""
    role = actor_role(cursor, actor_id)
    if role == ROLE_ADMIN:
        return role, None
    if role == ROLE_MANAGER:
        coop = manager_cooperative(cursor, actor_id)
        if coop is None:
            raise AssistanceError(403, "NO_COOPERATIVE", "Your manager account is not linked to a cooperative.")
        return role, coop["id"]
    raise AssistanceError(403, "FORBIDDEN", "You do not have permission for this action.")


def _group(cursor, actor_id, group_id, lock=False):
    role, coop = _scope(cursor, actor_id)
    try:
        group_id = int(group_id)
    except (TypeError, ValueError):
        raise AssistanceError(404, "NOT_FOUND", NOT_FOUND)
    cursor.execute("SELECT g.*, c.name AS cooperative FROM cooperative_groups g JOIN cooperatives c ON c.id = g.cooperative_id "
                   "WHERE g.id=%s" + (" FOR UPDATE" if lock else ""), (group_id,))
    g = cursor.fetchone()
    if g is None or (coop is not None and g["cooperative_id"] != coop):
        raise AssistanceError(404, "NOT_FOUND", NOT_FOUND)
    return role, g


def create_group(actor_id, name, cooperative_id=None):
    """Manager: always in their own cooperative (any cooperative_id sent is ignored). Admin: the given one."""
    name = clean_name(name)
    with transaction() as cursor:
        role, coop = _scope(cursor, actor_id)
        if coop is None:                                     # admin
            try:
                coop = int(cooperative_id)
            except (TypeError, ValueError):
                raise AssistanceError(400, "INVALID", "Choose a cooperative.")
            cursor.execute("SELECT id FROM cooperatives WHERE id=%s", (coop,))
            if cursor.fetchone() is None:
                raise AssistanceError(404, "NOT_FOUND", "Cooperative not found.")
        cursor.execute("SELECT id FROM cooperative_groups WHERE cooperative_id=%s AND name=%s", (coop, name))
        if cursor.fetchone():
            raise AssistanceError(409, "DUPLICATE", "This cooperative already has a group with that name.")
        now = utcnow()
        cursor.execute("INSERT INTO cooperative_groups (cooperative_id, name, created_by, created_at, updated_at) "
                       "VALUES (%s,%s,%s,%s,%s)", (coop, name, actor_id, now, now))
        group_id = cursor.lastrowid
        audit.record(audit.GROUP_CREATED, actor_id=actor_id, target_type="group", target_id=group_id,
                     cooperative_id=coop, details={"by": role}, cursor=cursor)
    return group_id


def update_group(actor_id, group_id, name=None, status=None):
    with transaction() as cursor:
        role, g = _group(cursor, actor_id, group_id, lock=True)
        new_name = clean_name(name) if name is not None else g["name"]
        new_status = status if status in ("ACTIVE", "INACTIVE") else g["status"]
        if new_name != g["name"]:
            cursor.execute("SELECT id FROM cooperative_groups WHERE cooperative_id=%s AND name=%s AND id<>%s",
                           (g["cooperative_id"], new_name, g["id"]))
            if cursor.fetchone():
                raise AssistanceError(409, "DUPLICATE", "This cooperative already has a group with that name.")
        cursor.execute("UPDATE cooperative_groups SET name=%s, status=%s, updated_at=%s WHERE id=%s",
                       (new_name, new_status, utcnow(), g["id"]))
        audit.record(audit.GROUP_UPDATED, actor_id=actor_id, target_type="group", target_id=g["id"],
                     cooperative_id=g["cooperative_id"],
                     details={"status": new_status, "renamed": new_name != g["name"], "by": role}, cursor=cursor)


def _member_in_cooperative(cursor, user_id, cooperative_id):
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise AssistanceError(404, "NOT_FOUND", "Member not found.")
    cursor.execute("SELECT m.user_id, m.group_id, u.role FROM cooperative_memberships m JOIN users u ON u.id = m.user_id "
                   "WHERE m.user_id=%s AND m.cooperative_id=%s AND m.member_role = u.role "
                   "AND u.role IN ('driver','umusare') AND m.status <> 'REVOKED' FOR UPDATE", (user_id, cooperative_id))
    row = cursor.fetchone()
    if row is None:                                   # unknown, or in another cooperative: same answer
        raise AssistanceError(404, "NOT_FOUND", "Member not found.")
    return row


def add_member(actor_id, group_id, user_id):
    with transaction() as cursor:
        role, g = _group(cursor, actor_id, group_id)
        if g["status"] != "ACTIVE":
            raise AssistanceError(409, "INACTIVE", "Activate the group before adding members.")
        m = _member_in_cooperative(cursor, user_id, g["cooperative_id"])
        if m["group_id"] == g["id"]:
            raise AssistanceError(409, "UNCHANGED", "This member is already in the group.")
        cursor.execute("UPDATE cooperative_memberships SET group_id=%s, group_assigned_by=%s, group_assigned_at=%s "
                       "WHERE user_id=%s AND cooperative_id=%s", (g["id"], actor_id, utcnow(), m["user_id"],
                                                                 g["cooperative_id"]))
        audit.record(audit.GROUP_MEMBER_ADDED, actor_id=actor_id, target_type="user", target_id=m["user_id"],
                     cooperative_id=g["cooperative_id"], details={"group": g["id"], "previous": m["group_id"]},
                     cursor=cursor)


def remove_member(actor_id, group_id, user_id):
    with transaction() as cursor:
        role, g = _group(cursor, actor_id, group_id)
        m = _member_in_cooperative(cursor, user_id, g["cooperative_id"])
        if m["group_id"] != g["id"]:
            raise AssistanceError(404, "NOT_FOUND", "Member not found in this group.")
        cursor.execute("UPDATE cooperative_memberships SET group_id=NULL, group_assigned_by=%s, group_assigned_at=%s "
                       "WHERE user_id=%s", (actor_id, utcnow(), m["user_id"]))
        audit.record(audit.GROUP_MEMBER_REMOVED, actor_id=actor_id, target_type="user", target_id=m["user_id"],
                     cooperative_id=g["cooperative_id"], details={"group": g["id"]}, cursor=cursor)


def groups_of(cursor, cooperative_id):
    cursor.execute("SELECT g.id, g.name, g.status, g.created_at, u.username AS created_by_name, "
                   "SUM(m.member_role='driver') AS drivers, SUM(m.member_role='umusare') AS umusare "
                   "FROM cooperative_groups g LEFT JOIN users u ON u.id = g.created_by "
                   "LEFT JOIN cooperative_memberships m ON m.group_id = g.id AND m.status <> 'REVOKED' "
                   "WHERE g.cooperative_id=%s GROUP BY g.id ORDER BY g.status, g.name", (cooperative_id,))
    rows = cursor.fetchall()
    for r in rows:
        r["code"] = group_code(r["id"])
        r["drivers"], r["umusare"] = int(r["drivers"] or 0), int(r["umusare"] or 0)
    return rows


def group_detail(actor_id, group_id):
    with transaction() as cursor:
        role, g = _group(cursor, actor_id, group_id)
        cursor.execute("SELECT u.id, u.username, u.email, u.role, u.is_active, m.group_id, "
                       "COALESCE(IF(u.role='driver', dp.verification_status, up.verification_status), 'PENDING') AS verification_status, "
                       "dp.vehicle_plate_number FROM cooperative_memberships m JOIN users u ON u.id = m.user_id "
                       "LEFT JOIN driver_profiles dp ON dp.user_id = u.id LEFT JOIN umusare_profiles up ON up.user_id = u.id "
                       "WHERE m.cooperative_id=%s AND m.member_role = u.role AND u.role IN ('driver','umusare') "
                       "AND m.status <> 'REVOKED' ORDER BY u.role, u.username", (g["cooperative_id"],))
        people = cursor.fetchall()
    for p in people:
        p["code"] = member_code(p["role"], p["id"])
    g["code"] = group_code(g["id"])
    return {"group": g, "members": [p for p in people if p["group_id"] == g["id"]],
            "candidates": [p for p in people if p["group_id"] != g["id"]], "role": role}


def active_groups_for_registration(cursor):
    """Active groups of approved cooperatives (names only) for the registration form."""
    cursor.execute("SELECT g.id, g.name, g.cooperative_id FROM cooperative_groups g JOIN cooperatives c "
                   "ON c.id = g.cooperative_id WHERE g.status='ACTIVE' AND c.status='APPROVED' ORDER BY g.name")
    return cursor.fetchall()
