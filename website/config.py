# website/config.py
"""Environment-based configuration. No secrets live in the code.

Values are read from environment variables; a local `.env` file (see
`.env.example`) is loaded first when python-dotenv is installed. Variables that
are already set in the environment always win over `.env`.

    SAFEDRIVE_ENV                 development (default) | production | testing
    SAFEDRIVE_SECRET_KEY          Flask session signing key (required in production)
    SAFEDRIVE_DEBUG               1 to enable the Flask debugger (never in production)
    SAFEDRIVE_SESSION_COOKIE_SECURE  1 when served over HTTPS
    SAFEDRIVE_DB_HOST / _PORT / _USER / _PASSWORD / _NAME   MySQL connection
"""
import logging
import os
import re
import secrets

log = logging.getLogger(__name__)

try:  # optional: plain environment variables work without it
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:  # pragma: no cover
    pass

DEFAULT_DB_NAME = "safedrive_ai_v2"
# The original database (corrupted tablespaces, old schema). It must never be opened by this app.
LEGACY_DB_NAMES = ("safedrive_ai",)


class ConfigError(RuntimeError):
    pass


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def db_settings():
    """MySQL connection settings, read at call time so tests can switch databases.

    Raises ConfigError for the legacy database, so no code path can connect to it.
    """
    name = os.environ.get("SAFEDRIVE_DB_NAME", DEFAULT_DB_NAME).strip()
    if name.lower() in LEGACY_DB_NAMES:
        raise ConfigError(f"database {name!r} is the retired legacy database and must not be used; "
                          f"set SAFEDRIVE_DB_NAME={DEFAULT_DB_NAME}")
    return {
        "host": os.environ.get("SAFEDRIVE_DB_HOST", "localhost"),
        "port": int(os.environ.get("SAFEDRIVE_DB_PORT", "3306")),
        "user": os.environ.get("SAFEDRIVE_DB_USER", "root"),
        "password": os.environ.get("SAFEDRIVE_DB_PASSWORD", ""),
        "database": name,
    }


DEFAULT_TILES = {
    # CARTO Voyager raster basemap (OpenStreetMap data); CARTO_API_KEY is appended when set. Attribution required.
    "url": "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
    "attribution": "&copy; OpenStreetMap contributors &copy; CARTO",
    "subdomains": "abcd",
    "max_zoom": 20,
}


def map_tiles():
    """Leaflet tile-layer settings for the browser.

    MAP_TILE_URL          tile URL template ({z}/{x}/{y}; optional {s}, {r}, {key})
    MAP_TILE_ATTRIBUTION  attribution text required by the provider
    MAP_TILE_SUBDOMAINS   letters substituted for {s} (empty if unused)
    MAP_TILE_MAX_ZOOM     highest zoom level the provider serves
    MAP_TILE_API_KEY      substituted for {key}; read from the environment only
    CARTO_API_KEY         CARTO Basemaps key, appended as ?key=... to CARTO (cartocdn.com) tile URLs

    The public tile.openstreetmap.org servers are not allowed for application use (they answer
    403 "Access blocked"), so they are never the default. An invalid URL falls back to the default.
    """
    url = os.environ.get("MAP_TILE_URL", "").strip()
    key = os.environ.get("MAP_TILE_API_KEY", "").strip()
    valid = (url.startswith("https://") and all(p in url for p in ("{z}", "{x}", "{y}"))
             and ("{key}" not in url or key))
    if url and not valid:
        log.warning("MAP_TILE_URL ignored (must be https with {z}/{x}/{y}, and {key} needs MAP_TILE_API_KEY)")
    if not (url and valid):
        return _with_carto_key(dict(DEFAULT_TILES))
    try:
        max_zoom = max(1, min(int(os.environ.get("MAP_TILE_MAX_ZOOM", "19")), 22))
    except ValueError:
        max_zoom = 19
    return _with_carto_key({"url": url.replace("{key}", key) if key else url,
                            "attribution": os.environ.get("MAP_TILE_ATTRIBUTION", "&copy; OpenStreetMap contributors").strip(),
                            "subdomains": os.environ.get("MAP_TILE_SUBDOMAINS", "abc").strip(),
                            "max_zoom": max_zoom})


_CARTO_KEY = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _with_carto_key(tiles):
    """Append CARTO_API_KEY (from .env / the environment) to CARTO basemap URLs as ?key=...

    The key is only ever read from the environment; it is never written into source files.
    Other providers are left untouched, and a URL that already carries a key is not changed.
    """
    key = os.environ.get("CARTO_API_KEY", "").strip()
    url = tiles["url"]
    if not key or "cartocdn.com" not in url or "key=" in url:
        return tiles
    if not _CARTO_KEY.match(key):
        log.warning("CARTO_API_KEY ignored: unexpected characters")
        return tiles
    tiles["url"] = f"{url}{'&' if '?' in url else '?'}key={key}"
    return tiles


def port_in_use(host, port, timeout=0.5):
    """True when something already accepts connections on host:port.

    On Windows a second development server can bind a port that is already in use
    (SO_REUSEADDR) without any error, and the browser keeps talking to the OLD server.
    main.py checks this before starting.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, int(port))) == 0


DEV_SECRET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instance", ".dev_secret_key")


def development_secret(path=None):
    """Random development-only key, created once in instance/.dev_secret_key (git-ignored) and reused.

    Falls back to a per-run random key if the file cannot be written.
    """
    path = path or DEV_SECRET_FILE
    try:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(key)
        log.warning("SAFEDRIVE_SECRET_KEY is not set; using the development key in instance/.dev_secret_key")
    except OSError:
        log.warning("SAFEDRIVE_SECRET_KEY is not set and instance/ is not writable; using a random key for this run")
    return key


def flask_config():
    """Flask settings for create_app(); raises ConfigError for unsafe production settings."""
    env = os.environ.get("SAFEDRIVE_ENV", "development").strip().lower()
    debug = env_flag("SAFEDRIVE_DEBUG")
    secret = os.environ.get("SAFEDRIVE_SECRET_KEY", "")

    if env == "production":
        problems = []
        if len(secret) < 32:
            problems.append("SAFEDRIVE_SECRET_KEY must be set (at least 32 characters)")
        if debug:
            problems.append("SAFEDRIVE_DEBUG must be off")
        if not db_settings()["password"]:
            problems.append("SAFEDRIVE_DB_PASSWORD must be set")
        if problems:
            raise ConfigError("Unsafe production configuration: " + "; ".join(problems))

    if not secret:
        if env == "development":
            # Stable across restarts (sessions and pending email codes stay valid); never used in production,
            # which refuses to start without SAFEDRIVE_SECRET_KEY (checked above).
            secret = development_secret()
        else:
            # testing: a fresh random key per app
            secret = secrets.token_hex(32)

    return {
        "ENV_NAME": env,
        "DEBUG": debug,
        "SECRET_KEY": secret,
        "TESTING": env == "testing",
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": env_flag("SAFEDRIVE_SESSION_COOKIE_SECURE", env == "production"),
        "REMEMBER_COOKIE_HTTPONLY": True,
        "REMEMBER_COOKIE_SAMESITE": "Lax",
        "WTF_CSRF_TIME_LIMIT": None,          # token valid for the whole session
        "MAX_CONTENT_LENGTH": 1 * 1024 * 1024,  # no uploads; forms are small
    }
