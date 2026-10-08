// Unread indicators (messages + notifications) and the notification menu. Plain HTTP polling.
(() => {
    const cfg = window.SAFEDRIVE_CHAT;
    if (!cfg) return;
    const meta = document.querySelector('meta[name="csrf-token"]');
    const token = meta ? meta.content : "";
    const bell = document.getElementById("notif-bell"), pop = document.getElementById("notif-pop");
    const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const POLL_MS = 20000;

    function setCount(kind, n) {
        document.querySelectorAll('[data-unread="' + kind + '"]').forEach(el => {
            el.textContent = n > 99 ? "99+" : String(n);
            el.hidden = !n;
        });
    }

    async function refresh() {
        if (document.hidden) return;
        try {
            const res = await fetch(cfg.unread, { cache: "no-store", credentials: "same-origin", headers: { Accept: "application/json" } });
            if (!res.ok || res.redirected) return;
            const d = await res.json();
            setCount("messages", d.messages || 0);
            setCount("notifications", d.notifications || 0);
            document.dispatchEvent(new CustomEvent("safedrive:unread", { detail: d }));
        } catch (e) { /* offline: keep the last known counts */ }
    }
    window.SafeDriveUnread = { refresh };

    async function openMenu() {
        pop.hidden = false; bell.setAttribute("aria-expanded", "true");
        pop.innerHTML = '<span class="empty">Loading…</span>';
        try {
            const res = await fetch(cfg.notifications, { cache: "no-store", credentials: "same-origin", headers: { Accept: "application/json" } });
            const d = await res.json();
            const items = d.notifications || [];
            pop.innerHTML = items.length ? items.map(n =>
                '<a role="menuitem" class="' + (n.unread ? "unread" : "") + '" href="' + esc(n.link || "#") + '">' + esc(n.title) +
                '<time>' + esc(new Date(n.created_at).toLocaleString()) + '</time></a>').join("")
                : '<span class="empty">No notifications yet.</span>';
            if (items.some(n => n.unread)) {
                await fetch(cfg.read, { method: "POST", credentials: "same-origin", headers: { "X-CSRFToken": token, Accept: "application/json" } });
                setCount("notifications", 0);
            }
        } catch (e) { pop.innerHTML = '<span class="empty">Could not load notifications. Try again.</span>'; }
    }
    function closeMenu() { pop.hidden = true; bell.setAttribute("aria-expanded", "false"); }

    if (bell && pop) {
        bell.addEventListener("click", ev => { ev.stopPropagation(); pop.hidden ? openMenu() : closeMenu(); });
        document.addEventListener("click", ev => { if (!pop.hidden && !pop.contains(ev.target)) closeMenu(); });
        document.addEventListener("keydown", ev => { if (ev.key === "Escape") closeMenu(); });
    }
    refresh();
    setInterval(refresh, POLL_MS);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
})();
