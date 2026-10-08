"""Map tile provider configuration (no public OSM tile servers, keys from the environment) and map fallback."""
import json
import os
import re

import pytest

from tests.web.conftest import login
from website.config import DEFAULT_TILES, map_tiles

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAP_JS = os.path.join(ROOT, "website", "static", "js", "assist-map.js")


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("MAP_TILE_URL", "MAP_TILE_ATTRIBUTION", "MAP_TILE_SUBDOMAINS", "MAP_TILE_MAX_ZOOM", "MAP_TILE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_provider_is_not_the_public_osm_tile_server(clean_env):
    t = map_tiles()
    assert t == DEFAULT_TILES
    assert "tile.openstreetmap.org" not in t["url"] and t["url"].startswith("https://")
    assert "OpenStreetMap contributors" in t["attribution"]                  # attribution kept
    assert "{key}" not in t["url"]                                           # no key needed or embedded


def test_provider_is_configurable_and_key_comes_from_environment(clean_env):
    clean_env.setenv("MAP_TILE_URL", "https://api.example-tiles.test/maps/streets/{z}/{x}/{y}.png?key={key}")
    clean_env.setenv("MAP_TILE_API_KEY", "env-secret-123")
    clean_env.setenv("MAP_TILE_ATTRIBUTION", "&copy; Example &copy; OpenStreetMap contributors")
    clean_env.setenv("MAP_TILE_SUBDOMAINS", "")
    clean_env.setenv("MAP_TILE_MAX_ZOOM", "18")
    t = map_tiles()
    assert t["url"] == "https://api.example-tiles.test/maps/streets/{z}/{x}/{y}.png?key=env-secret-123"
    assert (t["attribution"], t["subdomains"], t["max_zoom"]) == ("&copy; Example &copy; OpenStreetMap contributors", "", 18)


@pytest.mark.parametrize("url, key", [
    ("http://insecure.test/{z}/{x}/{y}.png", ""),                 # not https
    ("https://no-placeholders.test/tile.png", ""),                 # missing {z}/{x}/{y}
    ("https://needs-key.test/{z}/{x}/{y}.png?k={key}", ""),         # {key} without MAP_TILE_API_KEY
])
def test_invalid_configuration_falls_back_to_default(clean_env, url, key):
    clean_env.setenv("MAP_TILE_URL", url)
    if key:
        clean_env.setenv("MAP_TILE_API_KEY", key)
    assert map_tiles() == DEFAULT_TILES


def test_map_script_uses_configured_tiles_and_has_a_failure_fallback():
    js = open(MAP_JS, encoding="utf-8").read()
    assert "tile.openstreetmap.org/{z}" not in js and "SAFEDRIVE_TILES" in js
    assert 'referrerPolicy: "strict-origin-when-cross-origin"' in js     # origin only, never the page path
    assert '"tileerror"' in js and "showFallback" in js and "dataset.fallback" in js
    assert not re.search(r"(api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9]{8,}", js, re.I)   # no secrets


@pytest.mark.db
def test_dashboards_receive_tile_config_and_fallback_targets(app, db, clean_env):
    coop = db.cooperative()
    db.user("d@example.com", coop_id=coop)
    uid = db.user("u@example.com", role="umusare", coop_id=coop)
    db.query("INSERT INTO umusare_profiles (user_id, verification_status) VALUES (%s, 'VERIFIED')", (uid,))
    d, u = app.test_client(), app.test_client()
    login(d, "d@example.com")
    login(u, "u@example.com")
    for html in (d.get("/driver-dashboard").get_data(as_text=True), u.get("/umusare-dashboard").get_data(as_text=True)):
        m = re.search(r"window\.SAFEDRIVE_TILES = (\{.*?\});", html)
        assert m and json.loads(m.group(1))["url"] == DEFAULT_TILES["url"]
        assert "tile.openstreetmap.org" not in html
    dhtml = d.get("/driver-dashboard").get_data(as_text=True)
    assert 'data-fallback="map-fallback"' in dhtml and 'id="map-fallback"' in dhtml
    uhtml = u.get("/umusare-dashboard").get_data(as_text=True)
    assert 'data-fallback="map-active-fb"' in uhtml and 'id="map-active-fb"' in uhtml


@pytest.mark.db
def test_api_key_is_only_delivered_through_configuration(app, db, clean_env):
    clean_env.setenv("MAP_TILE_URL", "https://tiles.example.test/{z}/{x}/{y}.png?key={key}")
    clean_env.setenv("MAP_TILE_API_KEY", "from-env-only")
    db.user("d@example.com", coop_id=db.cooperative())
    c = app.test_client()
    login(c, "d@example.com")
    html = c.get("/driver-dashboard").get_data(as_text=True)
    assert "tiles.example.test/{z}/{x}/{y}.png?key=from-env-only" in html
    assert "from-env-only" not in open(MAP_JS, encoding="utf-8").read()


# ---------------------------------------------------------------- CARTO Basemaps API key
FAKE_KEY = "cb1_test_key_0000"


def test_carto_key_is_appended_from_environment(clean_env):
    clean_env.setenv("CARTO_API_KEY", FAKE_KEY)
    assert map_tiles()["url"] == DEFAULT_TILES["url"] + "?key=" + FAKE_KEY
    assert "CARTO" in map_tiles()["attribution"] and "OpenStreetMap" in map_tiles()["attribution"]


def test_carto_key_absent_or_invalid_leaves_url_unchanged(clean_env):
    clean_env.setenv("CARTO_API_KEY", "")
    assert map_tiles()["url"] == DEFAULT_TILES["url"]
    clean_env.setenv("CARTO_API_KEY", "bad key&x=1")
    assert map_tiles()["url"] == DEFAULT_TILES["url"]


def test_carto_key_only_added_to_carto_urls(clean_env):
    clean_env.setenv("CARTO_API_KEY", FAKE_KEY)
    clean_env.setenv("MAP_TILE_URL", "https://tiles.other.test/{z}/{x}/{y}.png")
    assert FAKE_KEY not in map_tiles()["url"]
    clean_env.setenv("MAP_TILE_URL", "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?lang=en")
    assert map_tiles()["url"].endswith("?lang=en&key=" + FAKE_KEY)


def test_real_key_is_not_in_any_committable_file():
    """The real key lives only in the git-ignored .env file."""
    import subprocess
    env_path = os.path.join(ROOT, ".env")
    if not os.path.isfile(env_path):
        pytest.skip("no local .env")
    real = next((line.split("=", 1)[1].strip() for line in open(env_path, encoding="utf-8")
                 if line.startswith("CARTO_API_KEY=")), "")
    if not real:
        pytest.skip("no CARTO_API_KEY in .env")
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
    assert ignored, ".env must be git-ignored"
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split("\n")
    source_types = (".py", ".html", ".js", ".css", ".md", ".txt", ".sql", ".json", ".ini", ".cfg", ".toml",
                    ".yml", ".yaml", ".example", ".csv")
    leaks = []
    for f in files:
        if not f or f.startswith(".venv/") or not f.lower().endswith(source_types):
            continue
        try:
            if real in open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore").read():
                leaks.append(f)
        except OSError:
            pass
    assert leaks == []
