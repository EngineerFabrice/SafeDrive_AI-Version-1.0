import logging
import os
import sys

from website import create_app, get_connection
from website.config import db_settings, port_in_use

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()

# Optional: test database connection at startup (same settings as the app)
try:
    get_connection(connect_timeout=5).close()
    print(f"[INFO] MySQL connection successful (database {db_settings()['database']})")
except Exception as e:
    print(f"[ERROR] Could not connect to MySQL: {e}")

if __name__ == "__main__":
    # Local development server on localhost:5000 (SAFEDRIVE_PORT overrides); debug only when SAFEDRIVE_DEBUG=1.
    # SAFEDRIVE_HOST=0.0.0.0 also listens on the LAN, so a physical phone running the mobile app can connect
    # (development only; see mobile/README.md). The default stays localhost-only.
    host, port = os.environ.get("SAFEDRIVE_HOST", "127.0.0.1").strip() or "127.0.0.1", int(os.environ.get("SAFEDRIVE_PORT", "5000"))
    if port_in_use("127.0.0.1" if host == "0.0.0.0" else host, port):
        sys.exit(f"[ERROR] http://{host}:{port} is already in use by another server (probably an older "
                 f"SafeDrive process that is still running old code). Stop it first, or set SAFEDRIVE_PORT.")
    app.run(debug=app.config["DEBUG"], host=host, port=port)
