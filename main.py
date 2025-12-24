from website import create_app
import pymysql

app = create_app()

# Optional: test database connection at startup
try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='safedrive_ai',
        port=3307
    )
    conn.close()
    print("[INFO] MySQL connection successful!")
except pymysql.err.OperationalError as e:
    print(f"[ERROR] Could not connect to MySQL: {e}")

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(debug=True, host="127.0.0.1", port=5000)
