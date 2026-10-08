"""Create the first SafeDrive AI administrator, or reset an administrator's password.

    python create_admin.py

Credentials come from SAFEDRIVE_ADMIN_EMAIL / SAFEDRIVE_ADMIN_PASSWORD when set,
otherwise they are asked for interactively. Nothing is hard-coded. Writes to the
configured database (default safedrive_ai_v2; run `python -m website.migrate` first).
"""
import getpass
import os
import sys

from website import bcrypt, get_connection
from website.auth import validate_password

ADMIN_NAME = "Administrator"


def main():
    email = (os.environ.get("SAFEDRIVE_ADMIN_EMAIL") or input("Admin email: ")).strip().lower()
    password = os.environ.get("SAFEDRIVE_ADMIN_PASSWORD") or getpass.getpass("Admin password: ")
    problems = validate_password(password)
    if "@" not in email or problems:
        sys.exit("Invalid admin credentials: " + "; ".join(problems or ["enter a valid email"]))

    hashed = bcrypt.generate_password_hash(password).decode("utf-8")
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, role FROM users WHERE email=%s", (email,))
            row = cursor.fetchone()
            if row and row["role"] != "admin":
                sys.exit(f"{email} exists with role {row['role']}; change it from the admin dashboard instead.")
            if row:
                cursor.execute("UPDATE users SET password_hash=%s, is_active=1 WHERE id=%s", (hashed, row["id"]))
                print("Administrator password updated.")
            else:
                cursor.execute("INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, 'admin')",
                               (ADMIN_NAME, email, hashed))
                print("Administrator created.")
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
