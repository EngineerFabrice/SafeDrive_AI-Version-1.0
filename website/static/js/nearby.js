// Nearby verified drivers (driver dashboard). Privacy model, enforced by the server:
// the server stores and returns only ~1 km grid cells, aggregated, for SafeDrive-verified drivers who opted in.
// Your own exact position is drawn only in your own browser ("YOU") and is never shown to anyone else.
// Location permission is requested only when you press the button, never repeatedly.
(() => {
    const root = document.getElementById("nearby");
    if (!root) return;
    const $ = id => document.getElementById(id);
    const token = document.querySelector('meta[name="csrf-token"]').content;
    const URL_NEARBY = root.dataset.url, URL_VIS = root.dataset.visibilityUrl, URL_PRESENCE = root.dataset.presenceUrl;
    const REFRESH_MS = 60000;
    let map = null, layer = null, timer = null, busy = false, granted = false;

    const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

    function state(kind, html, action) {
        const el = $("nearby-state");
        el.className = "map-state " + kind;
        el.innerHTML = '<div class="map-state-icon" aria-hidden="true">' + ({ loading: "◌", empty: "○", error: "!", denied: "⊘",
            info: "i", offline: "⌁" }[kind] || "i") + "</div><p>" + html + "</p>" +
            (action ? '<button type="button" class="btn btn-primary btn-sm" id="nearby-action">' + esc(action.label) + "</button>" : "");
        el.hidden = false;
        if (action) $("nearby-action").addEventListener("click", action.run);
    }
    function setCount(text) { $("nearby-count").textContent = text; }

    if (root.dataset.eligible !== "1") {
        setCount("Available after verification");
        state("info", "Nearby drivers are shown to SafeDrive-verified drivers. Complete your email, vehicle and cooperative " +
              "verification to use this feature.");
        return;
    }

    function ensureMap() {
        if (map || !window.L) return map;
        const t = window.SAFEDRIVE_TILES || {};
        if (!t.url) return null;
        $("nearby-map").hidden = false;
        map = window.L.map($("nearby-map"), { scrollWheelZoom: false, attributionControl: true }).setView([-1.9441, 30.0619], 13);
        window.L.tileLayer(t.url, { maxZoom: t.max_zoom || 19, attribution: t.attribution || "", subdomains: t.subdomains || "abc",
                                    referrerPolicy: "strict-origin-when-cross-origin" }).addTo(map);
        layer = window.L.layerGroup().addTo(map);
        return map;
    }

    function draw(me, data) {
        const m = ensureMap();
        if (!m) { $("nearby-map").hidden = true; return; }
        layer.clearLayers();
        const pts = [[me.latitude, me.longitude]];
        window.L.circleMarker([me.latitude, me.longitude], { radius: 9, color: "#fff", weight: 3, fillColor: "#1a73e8", fillOpacity: 1 })
            .addTo(layer).bindTooltip("YOU", { permanent: true, direction: "top", offset: [0, -8], className: "you-label" });
        for (const c of data.cells) {
            pts.push([c.lat, c.lon]);
            window.L.circle([c.lat, c.lon], { radius: 550, color: "#059669", weight: 1, fillColor: "#10b981", fillOpacity: 0.12 })
                .addTo(layer);
            const icon = window.L.divIcon({ className: "", iconSize: [34, 34], iconAnchor: [17, 17],
                html: '<span class="cell-pin" aria-label="' + c.count + ' nearby drivers">' + c.count + "</span>" });
            const names = c.drivers.map(d => "<li><b>" + esc(d.name) + '</b> <span class="vt">✓ Verified</span><br><span>' +
                esc(d.cooperative) + (d.group ? " · " + esc(d.group) : "") + "</span></li>").join("");
            const more = c.count > c.drivers.length ? "<li>+" + (c.count - c.drivers.length) + " more</li>" : "";
            window.L.marker([c.lat, c.lon], { icon }).addTo(layer).bindPopup(
                '<div class="cell-pop"><div class="k">NEARBY DRIVER' + (c.count > 1 ? "S" : "") + "</div><ul>" + names + more +
                "</ul><p>" + esc(c.distance_label) + " · approximate area (≈1 km)</p></div>");
        }
        if (pts.length > 1) m.fitBounds(window.L.latLngBounds(pts).pad(0.3), { maxZoom: 14 });
        else m.setView(pts[0], 13);
        setTimeout(() => m.invalidateSize(), 50);
    }

    function locate() {
        return new Promise((resolve, reject) => {
            navigator.geolocation.getCurrentPosition(p => resolve(p.coords), reject,
                { enableHighAccuracy: false, timeout: 15000, maximumAge: 60000 });
        });
    }

    async function refresh() {
        if (busy || document.hidden) return;
        if (!navigator.onLine) { state("offline", "You are offline. Nearby drivers will appear when your connection returns."); return; }
        busy = true;
        if (!map) state("loading", "Finding verified drivers near you…");
        let me;
        try { me = await locate(); granted = true; }
        catch (err) { busy = false; return denied(err); }
        const post = (url, body) => fetch(url, { method: "POST", credentials: "same-origin", cache: "no-store",
            headers: { "Content-Type": "application/json", Accept: "application/json", "X-CSRFToken": token },
            body: JSON.stringify(body || {}) });
        try {
            // 1) propose my position: the server keeps only the ~1 km cell, and only if the move is plausible.
            //    "too frequent" (429) or "implausible" (409) is fine: the lookup uses the server's last confirmed area.
            const up = await post(URL_PRESENCE, { lat: me.latitude, lon: me.longitude, accuracy: me.accuracy });
            if (up.status === 403) { const d = await up.json().catch(() => ({}));
                setCount("Not available"); state("denied", esc(d.error || "You do not have permission to view nearby drivers.")); return; }
            // 2) look up nearby drivers around the server-known position (no coordinates are sent)
            const res = await post(URL_NEARBY);
            const data = await res.json().catch(() => ({}));
            if (res.status === 429) { state("info", esc(data.error || "Please wait a moment.")); timer = setTimeout(refresh, REFRESH_MS); return; }
            if (res.status === 403) { setCount("Not available"); state("denied", esc(data.error || "You do not have permission to view nearby drivers.")); return; }
            if (!res.ok) throw new Error(data.error || "HTTP " + res.status);
            setCount(data.total ? data.total + (data.total === 1 ? " driver nearby" : " drivers nearby") : "No drivers nearby");
            $("nearby-radius").textContent = data.radius_km;
            draw(me, data);
            if (data.total) $("nearby-state").hidden = true;
            else state("empty", "No verified drivers are currently visible nearby.");
            clearTimeout(timer); timer = setTimeout(refresh, REFRESH_MS);
        } catch (e) {
            state("error", "Nearby drivers could not be loaded. Check your connection and try again.", { label: "Try again", run: refresh });
        } finally { busy = false; }
    }

    function denied(err) {
        setCount("Location needed");
        if (err && err.code === 1) {
            state("denied", "Location access is blocked. SafeDrive needs your location to show verified drivers near you; " +
                  "it is used to find your approximate area (≈1 km) and is never shown to other drivers exactly. " +
                  "Allow location for this site in your browser settings, then try again.", { label: "Try again", run: refresh });
        } else {
            state("error", "Location unavailable. Enable location to view nearby drivers.", { label: "Try again", run: refresh });
        }
    }

    function askFirst() {
        setCount("—");
        state("info", "See SafeDrive-verified drivers around you. Your location is used only to find your approximate area.",
              { label: "Show nearby drivers", run: refresh });
    }

    // Visibility opt-in (server-side setting)
    const toggle = $("nearby-visible");
    if (toggle) toggle.addEventListener("change", async () => {
        toggle.disabled = true;
        try {
            const res = await fetch(URL_VIS, { method: "POST", credentials: "same-origin",
                headers: { "Content-Type": "application/json", Accept: "application/json", "X-CSRFToken": token },
                body: JSON.stringify({ visible: toggle.checked }) });
            if (!res.ok) throw new Error();
            $("nearby-vis-note").textContent = toggle.checked
                ? "Other verified drivers can see your approximate area (≈1 km) while this page is open."
                : "You are hidden from other drivers.";
            if (granted) refresh();
        } catch (e) { toggle.checked = !toggle.checked; $("nearby-vis-note").textContent = "Could not save. Try again."; }
        finally { toggle.disabled = false; }
    });

    window.addEventListener("online", () => { if (granted) refresh(); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden && granted) refresh(); });

    if (!window.isSecureContext || !navigator.geolocation) {
        setCount("Location unavailable");
        state("error", "Location is not available in this browser or on this connection. Open SafeDrive over HTTPS (or on " +
              "127.0.0.1) to view nearby drivers.");
        return;
    }
    // Never prompt on page load: only continue automatically if permission was already granted.
    if (navigator.permissions && navigator.permissions.query) {
        navigator.permissions.query({ name: "geolocation" }).then(p => {
            if (p.state === "granted") refresh();
            else if (p.state === "denied") denied({ code: 1 });
            else askFirst();
        }).catch(askFirst);
    } else askFirst();
})();
