# -------------------- Imports --------------------
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin
import os
import pymysql

# -------------------- Flask Extensions --------------------
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = "routes.login"
login_manager.login_message_category = "info"

# -------------------- User Model --------------------
class User(UserMixin):
    def __init__(self, id, username, email, password, role='driver'):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.role = role

    def is_admin(self):
        return self.role == 'admin'

    def is_chef(self):
        return self.role == 'chef'

    def is_driver(self):
        return self.role == 'driver'

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"

# -------------------- Database Connection --------------------
def get_connection(connect_timeout=10):
    """Return a PyMySQL connection to the safedrive_ai database.

    Settings can be overridden with environment variables (SAFEDRIVE_DB_HOST,
    SAFEDRIVE_DB_PORT, SAFEDRIVE_DB_USER, SAFEDRIVE_DB_PASSWORD, SAFEDRIVE_DB_NAME);
    the defaults below are the project's original local settings.
    """
    return pymysql.connect(
        host=os.environ.get('SAFEDRIVE_DB_HOST', 'localhost'),
        user=os.environ.get('SAFEDRIVE_DB_USER', 'root'),
        password=os.environ.get('SAFEDRIVE_DB_PASSWORD', ''),
        database=os.environ.get('SAFEDRIVE_DB_NAME', 'safedrive_ai'),
        port=int(os.environ.get('SAFEDRIVE_DB_PORT', '3307')),
        connect_timeout=connect_timeout,
        cursorclass=pymysql.cursors.DictCursor
    )

# -------------------- Helper Functions --------------------
def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password, role FROM users WHERE email=%s",
        (email,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password=row['password'],
            role=row['role']
        )
    return None

def get_user_by_id(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password, role FROM users WHERE id=%s",
        (int(user_id),)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            password=row['password'],
            role=row['role']
        )
    return None

# -------------------- Application Factory --------------------
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'yoursecretkey'

    # Initialize extensions
    bcrypt.init_app(app)
    login_manager.init_app(app)

    # Flask-Login user loader
    @login_manager.user_loader
    def load_user(user_id):
        return get_user_by_id(user_id)

    # Register blueprint
    from .routes import routes
    app.register_blueprint(routes)

    # Real-time CV engine access (engine itself lives in the top-level `engine` package)
    from .monitoring import monitoring
    app.register_blueprint(monitoring)

    # Optional: print registered routes after blueprint registration
    print("\n[INFO] Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    print()

    return app
