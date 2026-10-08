# website/routes.py
"""Accounts, cooperatives and role dashboards (schema: website/migrations/0002_core_schema.sql)."""
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pymysql
from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user

from . import (MEMBER_ROLES, ROLE_ADMIN, ROLE_DRIVER, ROLE_MANAGER, ROLE_UMUSARE, ROLES,
               SELF_REGISTER_ROLES, bcrypt, get_connection, get_user_by_email)
from . import audit
from .assistance_service import AssistanceError
from .auth import dashboard_url, normalize_phone, roles_required, safe_local_path, validate_registration

routes = Blueprint("routes", __name__)

_COOP_CODE = re.compile(r"^[A-Z0-9-]{2,20}$")
_dummy_hash = None   # compared against when the email is unknown, so timing does not reveal accounts


@contextmanager
def db_cursor():
    """Cursor in one transaction: commit on success, roll back on any error."""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _membership(cursor, user_id):
    cursor.execute(
        "SELECT m.member_role, m.status AS membership_status, c.id AS cooperative_id, "
        "c.name AS cooperative_name, c.code AS cooperative_code "
        "FROM cooperative_memberships m JOIN cooperatives c ON c.id = m.cooperative_id "
        "WHERE m.user_id = %s", (user_id,))
    return cursor.fetchone()


def _approved_cooperatives(cursor):
    cursor.execute("SELECT id, name, code, district FROM cooperatives WHERE status='APPROVED' ORDER BY name")
    return cursor.fetchall()


def _safe_next(target):
    """Only allow same-site relative redirects after login."""
    return safe_local_path(target)


# ========================= HOME =========================
@routes.route('/')
def home():
    return render_template('home.html')


@routes.route('/dashboard')
@login_required
def dashboard():
    return redirect(dashboard_url(current_user))


# ========================= REGISTER =========================
@routes.route('/register', methods=['GET', 'POST'])
def register():
    """Create a driver / Umusare account: Terms accepted, then email OTP, then cooperative verification.

    The response never reveals whether an email is already registered: a duplicate gets the same
    "we sent you a code" screen (and the existing owner gets a notice email instead of a code).
    """
    from . import vehicle as vehicle_mod
    from . import email_otp
    from .email_otp import mask_email
    from .group_service import active_groups_for_registration
    from .legal import record_acceptance
    if current_user.is_authenticated:
        return redirect(dashboard_url(current_user))
    with db_cursor() as cursor:
        cooperatives = _approved_cooperatives(cursor)
        groups = active_groups_for_registration(cursor)

    form = {}
    if request.method == 'POST':
        keys = ("username", "email", "phone", "role", "cooperative_id", "group_id", "vehicle_plate_number",
                "vehicle_make", "vehicle_model", "vehicle_type")
        form = {k: (request.form.get(k) or "").strip() for k in keys}
        form["email"] = form["email"].lower()
        password = request.form.get("password") or ""
        errors = validate_registration(form["username"], form["email"], password,
                                       request.form.get("confirm_password") or "")
        if form["role"] not in SELF_REGISTER_ROLES:
            errors.append("Choose whether you register as a driver or as an Umusare.")
        coop_ids = {str(c["id"]): c for c in cooperatives}
        if form["cooperative_id"] not in coop_ids:
            errors.append("Choose your cooperative.")
        phone = None
        if form["phone"]:
            try:
                phone = normalize_phone(form["phone"])
            except ValueError as exc:
                errors.append(str(exc))
        group_id = None
        if form["group_id"]:
            match = [g for g in groups if str(g["id"]) == form["group_id"]
                     and str(g["cooperative_id"]) == form["cooperative_id"]]
            if match:
                group_id = match[0]["id"]
            else:
                errors.append("Choose a group of your own cooperative, or leave the group empty.")
        vehicle = None
        if form["role"] == ROLE_DRIVER:
            try:
                vehicle = vehicle_mod.validate(form, require_plate=False)
            except ValueError as exc:
                errors.append(str(exc))
        if request.form.get("accept_terms") != "1":
            errors.append("Please read and accept the Terms & Conditions and Privacy Policy to create an account.")
        if not errors and _registrations_from_this_network() >= _registration_limit():
            errors.append("Too many accounts were created from this network recently. Please try again later.")

        if not errors:
            hashed = bcrypt.generate_password_hash(password).decode('utf-8')
            masked = mask_email(form["email"])
            message = (f"We sent a 6-digit verification code to {masked}. "
                       "Enter it below to verify your email address.")
            try:
                with db_cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO users (username, email, password_hash, role, phone) VALUES (%s, %s, %s, %s, %s)",
                        (form["username"], form["email"], hashed, form["role"], phone))
                    user_id = cursor.lastrowid
                    coop_id = int(form["cooperative_id"])
                    now = _utcnow()
                    cursor.execute(
                        "INSERT INTO cooperative_memberships (user_id, cooperative_id, member_role, group_id, "
                        "group_assigned_at) VALUES (%s, %s, %s, %s, %s)",
                        (user_id, coop_id, form["role"], group_id, now if group_id else None))
                    if form["role"] == ROLE_DRIVER:
                        v = vehicle or {}
                        cursor.execute(
                            "INSERT INTO driver_profiles (user_id, vehicle_plate_number, vehicle_make, vehicle_model, "
                            "vehicle_type, vehicle_updated_at) VALUES (%s, %s, %s, %s, %s, %s)",
                            (user_id, v.get("vehicle_plate_number"), v.get("vehicle_make"), v.get("vehicle_model"),
                             v.get("vehicle_type"), now if v.get("vehicle_plate_number") else None))
                    else:
                        cursor.execute("INSERT INTO umusare_profiles (user_id) VALUES (%s)", (user_id,))
                    audit.record(audit.USER_REGISTERED, actor_id=user_id, target_type="user",
                                 target_id=user_id, cooperative_id=coop_id,
                                 details={"role": form["role"]}, cursor=cursor)
                    record_acceptance(cursor, user_id, now)
            except pymysql.err.IntegrityError:
                # Duplicate email: the response must be indistinguishable from a new account (same redirect,
                # message, cookie shape and countdown); the real owner gets a notice email, never a code.
                audit.record(audit.REGISTRATION_DUPLICATE, target_type="user")
                user_id = None
            state = email_otp.flow_new(user_id, form["email"])
            delivered = True
            try:
                if user_id:
                    email_otp.issue(user_id)
                else:
                    email_otp.decoy_send(form["email"])
            except AssistanceError as exc:
                delivered = False
                flash(exc.message, "warning")
            else:
                flash(message, "success")
            email_otp.flow_record_send(state, time.time(), delivered)   # counts even when delivery failed
            session["verify"] = {"t": email_otp.flow_dump(state), "masked": masked}
            return redirect(url_for('account.verify_email'))
        for e in errors:
            flash(e, "danger")
    return render_template('register.html', cooperatives=cooperatives, groups=groups, form=form,
                           vehicle_types=_vehicle_types())


def _vehicle_types():
    from .vehicle import VEHICLE_TYPES
    return VEHICLE_TYPES


def _registration_limit():
    try:
        return max(1, int(os.environ.get("REGISTRATION_LIMIT_PER_HOUR", "10")))
    except ValueError:
        return 10


def _registrations_from_this_network():
    """Accounts (and duplicate attempts) created from this IP address in the last hour."""
    with db_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS n FROM audit_logs WHERE action IN ('USER_REGISTERED','REGISTRATION_DUPLICATE') "
                       "AND ip_address <=> %s AND occurred_at >= %s",
                       (request.remote_addr, _utcnow() - timedelta(hours=1)))
        return cursor.fetchone()["n"]


# ========================= LOGIN =========================
@routes.route('/login', methods=['GET', 'POST'])
def login():
    global _dummy_hash
    if current_user.is_authenticated:
        return redirect(dashboard_url(current_user))
    if request.method == 'POST':
        email = (request.form.get('email') or "").strip().lower()
        password = request.form.get('password') or ""
        user = get_user_by_email(email) if email else None

        if user is None:
            if _dummy_hash is None:
                _dummy_hash = bcrypt.generate_password_hash("timing-equaliser").decode()
            bcrypt.check_password_hash(_dummy_hash, password)
            ok = False
        else:
            ok = user.is_active and bcrypt.check_password_hash(user.password_hash, password)

        if ok:
            login_user(user)
            with db_cursor() as cursor:
                cursor.execute("UPDATE users SET last_login_at=%s WHERE id=%s", (_utcnow(), user.id))
            audit.record(audit.LOGIN_SUCCEEDED, actor_id=user.id, target_type="user", target_id=user.id)
            return redirect(_safe_next(request.args.get("next")) or dashboard_url(user))

        audit.record(audit.LOGIN_FAILED, target_type="user", target_id=user.id if user else None)
        flash('Incorrect email or password.', 'danger')
    return render_template('login.html')


# ========================= LOGOUT =========================
@routes.route('/logout', methods=['POST'])
@login_required
def logout():
    audit.record(audit.LOGOUT, actor_id=current_user.id, target_type="user", target_id=current_user.id)
    logout_user()
    flash('You have been signed out.', 'info')
    return redirect(url_for('routes.home'))


# ========================= ADMIN =========================
@routes.route('/admin-dashboard')
@roles_required(ROLE_ADMIN)
def admin_dashboard():
    with db_cursor() as cursor:
        cursor.execute(
            "SELECT u.id, u.username, u.email, u.role, u.is_active, m.status AS membership_status, "
            "c.id AS cooperative_id, c.name AS cooperative_name "
            "FROM users u LEFT JOIN cooperative_memberships m ON m.user_id = u.id "
            "LEFT JOIN cooperatives c ON c.id = m.cooperative_id ORDER BY u.created_at DESC")
        users = cursor.fetchall()
        cursor.execute(
            "SELECT c.id, c.name, c.code, c.district, c.status, "
            "SUM(m.member_role='driver' AND m.status='APPROVED') AS drivers, "
            "SUM(m.member_role='umusare' AND m.status='APPROVED') AS umusare, "
            "SUM(m.status='PENDING') AS pending "
            "FROM cooperatives c LEFT JOIN cooperative_memberships m ON m.cooperative_id = c.id "
            "GROUP BY c.id ORDER BY c.name")
        cooperatives = cursor.fetchall()

    counts = {role: sum(1 for u in users if u['role'] == role) for role in ROLES}
    from .assistance_service import admin_overview
    return render_template('admin-dashboard.html', users=users, cooperatives=cooperatives,
                           counts=counts, roles=ROLES, assistance=admin_overview())


@routes.route('/admin/cooperatives', methods=['POST'])
@roles_required(ROLE_ADMIN)
def create_cooperative():
    """Create a cooperative. With a manager it is ACTIVE; without one it is kept as an inactive draft
    (not offered at registration) until an administrator assigns a manager and activates it."""
    back = request.form.get("next") or ""
    back = back if back.startswith("/admin") and not back.startswith("//") else url_for('routes.admin_dashboard')
    name = (request.form.get('name') or "").strip()
    code = (request.form.get('code') or "").strip().upper()
    district = (request.form.get('district') or "").strip() or None
    manager_raw = (request.form.get('manager_id') or "").strip()
    if not 3 <= len(name) <= 120 or not _COOP_CODE.match(code) or (district and len(district) > 80):
        flash("Enter a name (3–120 characters) and a code of 2–20 letters, digits or dashes.", "danger")
        return redirect(back)
    try:
        with db_cursor() as cursor:
            cursor.execute("INSERT INTO cooperatives (name, code, district, status, created_by) "
                           "VALUES (%s, %s, %s, 'SUSPENDED', %s)", (name, code, district, current_user.id))
            coop_id = cursor.lastrowid
            audit.record(audit.COOPERATIVE_CREATED, actor_id=current_user.id, target_type="cooperative",
                         target_id=coop_id, cooperative_id=coop_id, details={"code": code}, cursor=cursor)
    except pymysql.err.IntegrityError:
        flash("A cooperative with this name or code already exists.", "danger")
        return redirect(back)
    if not manager_raw:
        flash(f"Cooperative {name} saved as an inactive draft. Assign a manager, then activate it.", "warning")
        return redirect(url_for('coop.cooperative', coop_id=coop_id))
    from .cooperative_service import assign_manager, update_cooperative
    try:
        assign_manager(int(current_user.id), coop_id, manager_raw)
        update_cooperative(int(current_user.id), coop_id, name, district, "APPROVED")
    except AssistanceError as exc:
        flash(f"Cooperative {name} saved as an inactive draft: {exc.message}", "warning")
    else:
        flash(f"Cooperative {name} created and active.", "success")
    return redirect(url_for('coop.cooperative', coop_id=coop_id))


def _other_admins(cursor, user_id):
    cursor.execute("SELECT COUNT(*) AS n FROM users WHERE role='admin' AND is_active=1 AND id<>%s", (user_id,))
    return cursor.fetchone()["n"]


@routes.route('/admin/update-member', methods=['POST'])
@roles_required(ROLE_ADMIN)
def update_member():
    """Set a user's role and cooperative. Admin assignment approves the membership."""
    try:
        user_id = int(request.form.get('user_id', ''))
    except ValueError:
        user_id = None
    new_role = request.form.get('role')
    coop_raw = request.form.get('cooperative_id') or ""

    if user_id is None or new_role not in ROLES:
        flash("Invalid request.", "danger")
        return redirect(url_for('routes.admin_dashboard'))
    if user_id == int(current_user.id):
        flash("You cannot change your own role.", "warning")
        return redirect(url_for('routes.admin_dashboard'))

    with db_cursor() as cursor:
        cursor.execute("SELECT id, role FROM users WHERE id=%s FOR UPDATE", (user_id,))
        target = cursor.fetchone()
        if target is None:
            flash("User not found.", "danger")
            return redirect(url_for('routes.admin_dashboard'))
        if target["role"] == ROLE_ADMIN and new_role != ROLE_ADMIN and _other_admins(cursor, user_id) == 0:
            flash("At least one administrator must remain.", "warning")
            return redirect(url_for('routes.admin_dashboard'))

        coop_id = None
        if new_role in MEMBER_ROLES:
            cursor.execute("SELECT id FROM cooperatives WHERE id=%s AND status='APPROVED'",
                           (int(coop_raw) if coop_raw.isdigit() else -1,))
            row = cursor.fetchone()
            if row is None:
                flash("Drivers, Umusare and managers must be assigned to an approved cooperative.", "danger")
                return redirect(url_for('routes.admin_dashboard'))
            coop_id = row["id"]
            cursor.execute(
                "INSERT INTO cooperative_memberships (user_id, cooperative_id, member_role, status, "
                "reviewed_by, reviewed_at) VALUES (%s, %s, %s, 'APPROVED', %s, %s) "
                "ON DUPLICATE KEY UPDATE group_id=IF(cooperative_id=VALUES(cooperative_id), group_id, NULL), "
                "cooperative_id=VALUES(cooperative_id), member_role=VALUES(member_role), "
                "status='APPROVED', reviewed_by=VALUES(reviewed_by), reviewed_at=VALUES(reviewed_at)",
                (user_id, coop_id, new_role, current_user.id, _utcnow()))
            if new_role == ROLE_DRIVER:
                cursor.execute("INSERT IGNORE INTO driver_profiles (user_id) VALUES (%s)", (user_id,))
            elif new_role == ROLE_UMUSARE:
                # New Umusare start unverified; verification is a separate manager decision.
                cursor.execute("INSERT IGNORE INTO umusare_profiles (user_id) VALUES (%s)", (user_id,))
        else:
            # Administrators are platform staff, not cooperative members.
            cursor.execute("UPDATE cooperative_memberships SET status='REVOKED', reviewed_by=%s, reviewed_at=%s "
                           "WHERE user_id=%s", (current_user.id, _utcnow(), user_id))

        cursor.execute("UPDATE users SET role=%s WHERE id=%s", (new_role, user_id))
        audit.record(audit.ROLE_CHANGED, actor_id=current_user.id, target_type="user", target_id=user_id,
                     cooperative_id=coop_id, details={"from": target["role"], "to": new_role}, cursor=cursor)

    flash(f"User updated: role {new_role}.", "success")
    return redirect(url_for('routes.admin_dashboard'))


@routes.route('/admin/delete-user', methods=['POST'])
@roles_required(ROLE_ADMIN)
def delete_user():
    try:
        user_id = int(request.form.get('user_id', ''))
    except ValueError:
        user_id = None
    if user_id is None or user_id == int(current_user.id):
        flash("Invalid request or you cannot delete yourself.", "danger")
        return redirect(url_for('routes.admin_dashboard'))

    with db_cursor() as cursor:
        cursor.execute("SELECT role FROM users WHERE id=%s FOR UPDATE", (user_id,))
        target = cursor.fetchone()
        if target is None:
            flash("User not found.", "danger")
            return redirect(url_for('routes.admin_dashboard'))
        if target["role"] == ROLE_ADMIN and _other_admins(cursor, user_id) == 0:
            flash("At least one administrator must remain.", "warning")
            return redirect(url_for('routes.admin_dashboard'))
        audit.record(audit.USER_DELETED, actor_id=current_user.id, target_type="user", target_id=user_id,
                     details={"role": target["role"]}, cursor=cursor)
        cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))

    flash("User removed.", "success")
    return redirect(url_for('routes.admin_dashboard'))


# ========================= MANAGER =========================
@routes.route('/manager-dashboard')
@roles_required(ROLE_MANAGER)
def manager_dashboard():
    """Operational console for the manager's OWN cooperative only (scope from the database)."""
    from .cooperative_service import manager_console
    with db_cursor() as cursor:
        membership = _membership(cursor, current_user.id)
    console = manager_console(current_user.id)
    return render_template('manager-dashboard.html', membership=membership, console=console)


# ========================= DRIVER =========================
@routes.route('/driver-dashboard')
@roles_required(ROLE_DRIVER)
def driver_dashboard():
    from .driver_views import dashboard_context
    with db_cursor() as cursor:
        membership = _membership(cursor, current_user.id)
    ctx = dashboard_context(current_user.id)
    return render_template('driver-dashboard.html', membership=membership, vstate=ctx["vstate"], ctx=ctx)


# ========================= UMUSARE =========================
@routes.route('/umusare-dashboard')
@roles_required(ROLE_UMUSARE)
def umusare_dashboard():
    with db_cursor() as cursor:
        membership = _membership(cursor, current_user.id)
        cursor.execute("SELECT verification_status, availability FROM umusare_profiles WHERE user_id=%s",
                       (current_user.id,))
        profile = cursor.fetchone()
        cursor.execute("SELECT phone FROM users WHERE id=%s", (current_user.id,))
        phone = (cursor.fetchone() or {}).get("phone")
    from .badges import for_user
    from .cooperative_service import own_verification
    return render_template('umusare-dashboard.html', membership=membership, profile=profile, current_user_phone=phone,
                           vstate=own_verification(current_user.id), badge=for_user(current_user.id))
