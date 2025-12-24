# -------------------- Imports --------------------
from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin
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
def get_connection():
    """Return a PyMySQL connection to the safedrive_ai database"""
    return pymysql.connect(
        host='localhost',
        user='root',          # MySQL user
        password='',          # MySQL password (empty for no password)
        database='safedrive_ai',
        port=3307,             # <-- specify your custom port
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

    # Optional: print registered routes after blueprint registration
    print("\n[INFO] Registered routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    print()

    return app
