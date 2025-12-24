from website import get_connection, bcrypt

# ---------------- Admin credentials ----------------
admin_email = "admin@safedrive.ai"
admin_password = "12345"

# ---------------- Generate hashed password ----------------
hashed_password = bcrypt.generate_password_hash(admin_password).decode('utf-8')

# ---------------- Connect to DB ----------------
conn = get_connection()
cursor = conn.cursor()

# Check if admin already exists
cursor.execute("SELECT id FROM users WHERE email=%s", (admin_email,))
if cursor.fetchone():
    print("Admin already exists. Updating password...")
    cursor.execute(
        "UPDATE users SET password=%s WHERE email=%s",
        (hashed_password, admin_email)
    )
else:
    cursor.execute(
        "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
        ("Admin", admin_email, hashed_password, "admin")
    )

conn.commit()
cursor.close()
conn.close()

print("✅ Admin user created or password updated successfully!")
