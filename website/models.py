from flask_login import UserMixin
from . import login_manager, get_connection

class User(UserMixin):
    def __init__(self, id, username, email, password, role='driver'):
        self.id = id
        self.username = username
        self.email = email
        self.password = password
        self.role = role

    # ------------------ Role Checks ------------------
    def is_admin(self):
        return self.role == 'admin'

    def is_chef(self):
        return self.role == 'chef'

    def is_driver(self):
        return self.role == 'driver'

    # ------------------ Flask-Login ------------------
    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"

# ------------------ User Loader ------------------
@login_manager.user_loader
def load_user(user_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password, role FROM users WHERE id=%s",
        (user_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return User(id=row['id'], username=row['username'], email=row['email'], password=row['password'], role=row['role'])
    return None

# ------------------ Helper Functions ------------------
def get_all_users():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password, role FROM users")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [User(id=row['id'], username=row['username'], email=row['email'], password=row['password'], role=row['role']) for row in rows]

def get_users_by_role(role):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password, role FROM users WHERE role=%s", (role,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [User(id=row['id'], username=row['username'], email=row['email'], password=row['password'], role=row['role']) for row in rows]

def get_user_by_email(email):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, password, role FROM users WHERE email=%s", (email,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        return User(id=row['id'], username=row['username'], email=row['email'], password=row['password'], role=row['role'])
    return None
