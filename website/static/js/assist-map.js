/* SafeDrive assistance map (Leaflet; tile provider configured server-side, see MAP_TILE_* settings).
 *
 * The server decides what each user may see; this helper only draws what the API returned:
 *   approx  -> ~1 km circle (before acceptance: never an exact point)
 *   driver  -> blue marker   (only sent to the accepted Umusare / the driver themself)
 *   umusare -> green marker  (only sent to the driver of the accepted request)
 * If Leaflet cannot load (offline, blocked CDN), `available` is false and the page shows its
 * text fallback (status, distance, contact, navigation link); the workflow never depends on the map.
 */
(function () {
    const COLORS = { driver: "#2563eb", umusare: "#16a34a", approx: "#f59e0b" };
    const LABELS = { driver: "📍 Driver", umusare: "📍 Umusare" };

    // Pages name their text fallback with data-fallback="<element id>".
    function showFallback(el) {
        const fb = el.dataset.fallback && document.getElementById(el.dataset.fallback);
        if (fb) fb.hidden = false;
    }

    function SafeMap(el) {
        if (!el) return { available: false, update() {}, destroy() {} };
        if (!window.L) {
            el.hidden = true;
            return { available: false, update() {}, destroy() {} };
        }
        el.hidden = false;
        const map = window.L.map(el, { zoomControl: true, attributionControl: true, scrollWheelZoom: false })
            .setView([-1.9441, 30.0619], 13);
        // Tile provider comes from server configuration (website/config.py map_tiles); never the public
        // tile.openstreetmap.org servers, which block application traffic with HTTP 403.
        const t = window.SAFEDRIVE_TILES || {};
        if (!t.url) {
            el.hidden = true; showFallback(el);
            return { available: false, failed: true, update() {}, destroy() { map.remove(); } };
        }
        const tiles = window.L.tileLayer(t.url, {
            maxZoom: t.max_zoom || 19, attribution: t.attribution || "", subdomains: t.subdomains || "abc",
            // The app sends "Referrer-Policy: same-origin"; tile images send only the site origin, which
            // providers use to authorise keys. Paths and query strings are never sent.
            referrerPolicy: "strict-origin-when-cross-origin",
        }).addTo(map);
        // If the provider is unreachable or refuses us, fall back to the page's text information.
        let loaded = 0, failed = 0;
        tiles.on("tileload", () => { loaded += 1; });
        tiles.on("tileerror", () => {
            failed += 1;
            if (!loaded && failed >= 4) { api.failed = true; el.hidden = true; showFallback(el); }
        });
        const layers = {};
        let fitted = "";

        function put(kind, pos) {
            if (!pos) { if (layers[kind]) { map.removeLayer(layers[kind]); delete layers[kind]; } return; }
            const ll = [pos.lat, pos.lon];
            if (kind === "approx") {
                if (!layers.approx) {
                    layers.approx = window.L.circle(ll, { radius: 1000, color: COLORS.approx, weight: 2, dashArray: "6 6",
                        fillOpacity: 0.12 }).addTo(map).bindTooltip("Approximate area (±1 km)");
                } else layers.approx.setLatLng(ll);
                return;
            }
            if (!layers[kind]) {
                layers[kind] = window.L.circleMarker(ll, { radius: 10, color: "#fff", weight: 3, fillColor: COLORS[kind],
                    fillOpacity: 1 }).addTo(map).bindTooltip(LABELS[kind], { permanent: true, direction: "top", offset: [0, -10] });
            } else layers[kind].setLatLng(ll);
        }

        const api = {
            available: true, failed: false,
            update(points) {
                ["approx", "driver", "umusare"].forEach(k => put(k, points[k]));
                const keys = Object.keys(layers).sort().join(",");
                if (keys && keys !== fitted) {          // re-frame only when the set of markers changes
                    const group = window.L.featureGroup(Object.values(layers));
                    map.fitBounds(group.getBounds().pad(0.4), { maxZoom: 16 });
                    fitted = keys;
                }
                setTimeout(() => map.invalidateSize(), 50);
            },
            destroy() { map.remove(); },
        };
        return api;
    }

    window.SafeMap = SafeMap;
})();
