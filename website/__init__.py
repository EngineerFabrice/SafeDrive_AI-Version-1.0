# -------------------- Imports --------------------
import logging

import pymysql
from flask import Flask, jsonify, render_template, request
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin
from flask_wtf.csrf import CSRFError, CSRFProtect

from .config import db_settings, flask_config

log = logging.getLogger(__name__)

# -------------------- Flask Extensions --------------------
bcrypt = Bcrypt()
csrf = CSRFProtect()
login_manager = LoginManager()
login_manager.login_view = "routes.login"
login_manager.login_message_category = "info"
login_manager.session_protection = "strong"

# -------------------- Roles --------------------
ROLE_DRIVER = "driver"
ROLE_UMUSARE = "umusare"
ROLE_MANAGER = "manager"
ROLE_ADMIN = "admin"
ROLES = (ROLE_DRIVER, ROLE_UMUSARE, ROLE_MANAGER, ROLE_ADMIN)
MEMBER_ROLES = (ROLE_DRIVER, ROLE_UMUSARE, ROLE_MANAGER)   # roles that belong to a cooperative
SELF_REGISTER_ROLES = (ROLE_DRIVER, ROLE_UMUSARE)           # managers/admins are appointed

USER_COLUMNS = "id, username, email, password_hash, role, is_active, email_verified_at, terms_version, privacy_version"


# -------------------- User Model --------------------
class User(UserMixin):
    def __init__(self, id, username, email, password_hash, role=ROLE_DRIVER, active=True,
                 email_verified_at=None, terms_version=None, privacy_version=None):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role
        self.active = bool(active)
        self.email_verified_at = email_verified_at      # set only by a successful email OTP
        self.terms_version = terms_version              # currently accepted Terms / Privacy versions
        self.privacy_version = privacy_version

    @classmethod
    def from_row(cls, row):
        return cls(id=row["id"], username=row["username"], email=row["email"],
                   password_hash=row["password_hash"], role=row["role"], active=row["is_active"],
                   email_verified_at=row.get("email_verified_at"), terms_version=row.get("terms_version"),
                   privacy_version=row.get("privacy_version"))

    @property
    def email_verified(self):
        return self.email_verified_at is not None

    @property
    def is_active(self):
        return self.active

    def is_admin(self):
        return self.role == ROLE_ADMIN

    def is_manager(self):
        return self.role == ROLE_MANAGER

    def is_driver(self):
        return self.role == ROLE_DRIVER

    def is_umusare(self):
        return self.role == ROLE_UMUSARE

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# -------------------- Database Connection --------------------
def get_connection(connect_timeout=10):
    """Return a PyMySQL connection using the environment settings (see website/config.py)."""
    return pymysql.connect(**db_settings(), connect_timeout=connect_timeout,
                           cursorclass=pymysql.cursors.DictCursor)


# -------------------- Helper Functions --------------------
def _fetch_user(where, value):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT {USER_COLUMNS} FROM users WHERE {where}=%s", (value,))
            row = cursor.fetchone()
    finally:
        conn.close()
    return User.from_row(row) if row else None


def get_user_by_email(email):
    return _fetch_user("email", email.strip().lower())


def get_user_by_id(user_id):
    try:
        return _fetch_user("id", int(user_id))
    except (TypeError, ValueError):
        return None


# -------------------- Application Factory --------------------
def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(flask_config())
    if overrides:
        app.config.update(overrides)

    # Initialize extensions
    bcrypt.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)

    # Flask-Login user loader
    @login_manager.user_loader
    def load_user(user_id):
        user = get_user_by_id(user_id)
        return user if user and user.is_active else None

    # Register blueprint
    from .routes import routes
    app.register_blueprint(routes)

    # Real-time CV engine access (engine itself lives in the top-level `engine` package)
    from .monitoring import monitoring
    app.register_blueprint(monitoring)

    # Driver -> Umusare assistance workflow
    from .assistance import assistance
    app.register_blueprint(assistance)

    # Admin management (drivers, Umusare, assistance monitoring, pricing)
    from .admin_views import admin_mgmt
    app.register_blueprint(admin_mgmt)

    # Cooperative management / member verification, and the internal chat
    from .cooperative_views import coop
    app.register_blueprint(coop)
    from .chat import chat
    app.register_blueprint(chat)

    # Email verification / legal pages, and driver-only vehicle + nearby-driver endpoints
    from .account import account
    app.register_blueprint(account)
    from .driver_views import driver
    app.register_blueprint(driver)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    def _wants_json():
        monitoring_api = request.path.startswith("/monitoring/") and request.path != "/monitoring/"
        return monitoring_api or request.path.startswith(("/assistance/", "/chat/api/", "/driver/nearby", "/driver/presence"))             or request.accept_mimetypes.best == "application/json"

    @app.errorhandler(CSRFError)
    def csrf_error(err):
        if _wants_json():
            return jsonify({"error": "Invalid or missing CSRF token."}), 400
        return render_template("error.html", code=400,
                               message="Your session form expired. Please go back and try again."), 400

    @app.errorhandler(403)
    def forbidden(err):
        if _wants_json():
            return jsonify({"error": "You do not have permission for this action."}), 403
        return render_template("error.html", code=403,
                               message="You do not have permission to open this page."), 403

    @app.errorhandler(404)
    def not_found(err):
        if _wants_json():
            return jsonify({"error": "Not found."}), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    log.debug("Registered routes: %s", ", ".join(sorted(str(r) for r in app.url_map.iter_rules())))
    return app
