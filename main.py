from website import create_app, get_connection
import pymysql

app = create_app()

# Optional: test database connection at startup (same settings as the app)
try:
    conn = get_connection(connect_timeout=5)
    conn.close()
    print("[INFO] MySQL connection successful!")
except pymysql.err.OperationalError as e:
    print(f"[ERROR] Could not connect to MySQL: {e}")

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(debug=True, host="127.0.0.1", port=5000)
