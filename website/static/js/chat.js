// SafeDrive internal chat: conversation list, contact search, message panel. Plain HTTP polling.
// The server authorizes every call; this script only renders what it is allowed to see.
(() => {
    const root = document.getElementById("chat");
    if (!root) return;
    const $ = id => document.getElementById(id);
    const token = document.querySelector('meta[name="csrf-token"]').content;
    const API = root.dataset.api, CONTACTS = root.dataset.contacts, MANAGER = root.dataset.manager;
    const LIST_MS = 15000, OPEN_MS = 5000;
    let conversations = [], current = null, lastId = 0, openTimer = null, searchTimer = null, sending = false;

    const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    const when = iso => {
        if (!iso) return "";
        const d = new Date(iso), now = new Date();
        return d.toDateString() === now.toDateString()
            ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
            : d.toLocaleDateString([], { day: "numeric", month: "short" });
    };
    const roleBadge = p => '<span class="role-badge role-' + esc(p.role) + '">' + esc(p.role_label) + "</span>";
    const metaLine = p => '<div class="meta-line">' + roleBadge(p) + (p.cooperative ? "<span>" + esc(p.cooperative) + "</span>" : "")
        + (p.verified === true ? '<span class="badge ok">✓ Verified</span>' : p.verified === false ? '<span class="badge warn">Not verified</span>' : "")
        + (p.email ? "<span>" + esc(p.email) + "</span>" : "") + "</div>";

    async function api(url, method = "GET", body) {
        let res;
        try {
            res = await fetch(url, { method, cache: "no-store", credentials: "same-origin",
                headers: { Accept: "application/json", "Content-Type": "application/json", "X-CSRFToken": token },
                body: body ? JSON.stringify(body) : undefined });
        } catch (e) { throw new Error("Cannot reach SafeDrive. Check your connection and try again."); }
        if (res.redirected && res.url.includes("/login")) throw new Error("Your session has expired. Please sign in again.");
        const data = await res.json().catch(() => null);
        if (!res.ok) {
            const msg = data && data.error;
            if (msg && /csrf/i.test(msg)) throw new Error("Security check failed. Reload the page and try again.");
            throw new Error(msg || "Request failed (HTTP " + res.status + ").");
        }
        return data || {};
    }

    // ------------------------------------------------------------ conversation list
    function renderList() {
        const q = $("conv-search").value.trim().toLowerCase();
        const items = conversations.filter(c => !q || (c.with.name + " " + (c.with.cooperative || "") + " " + c.with.role_label)
            .toLowerCase().includes(q));
        const list = $("conv-list");
        if (!conversations.length) {
            list.innerHTML = '<li class="state">No conversations yet. Use “New conversation” to contact someone.</li>';
            return;
        }
        if (!items.length) { list.innerHTML = '<li class="state">No conversation matches your search.</li>'; return; }
        list.innerHTML = items.map(c =>
            '<li><button type="button" data-id="' + esc(c.id) + '" class="' + (current === c.id ? "on" : "") + '">'
            + '<span class="row1"><b>' + esc(c.with.name) + '</b><span class="when">' + esc(when(c.last_at)) + "</span></span>"
            + metaLine(c.with)
            + '<span class="row1"><span class="preview">' + esc(c.last_message || "No messages yet") + "</span>"
            + (c.unread ? '<span class="unread-pill" aria-label="' + c.unread + ' unread">' + c.unread + "</span>" : "") + "</span>"
            + "</button></li>").join("");
    }

    async function loadList() {
        try {
            conversations = (await api(API)).conversations || [];
            $("side-error").textContent = "";
            renderList();
        } catch (e) {
            $("side-error").textContent = e.message;
            if (!conversations.length) $("conv-list").innerHTML = '<li class="state">Conversations could not be loaded.</li>';
        }
    }

    // ------------------------------------------------------------ contacts (new conversation)
    async function searchContacts() {
        const list = $("contact-list");
        list.hidden = false; $("conv-list").hidden = true;
        list.innerHTML = '<li class="state">Searching…</li>';
        try {
            const q = $("conv-search").value.trim();
            const people = (await api(CONTACTS + "?q=" + encodeURIComponent(q))).contacts || [];
            list.innerHTML = people.length ? people.map(p =>
                '<li><button type="button" data-contact="' + esc(p.token) + '"><b>' + esc(p.name) + "</b>" + metaLine(p) + "</button></li>").join("")
                : '<li class="state">No one you can contact matches this search.</li>';
        } catch (e) { list.innerHTML = '<li class="state">' + esc(e.message) + "</li>"; }
    }
    function closeContacts() { $("contact-list").hidden = true; $("conv-list").hidden = false; $("new-btn").textContent = "New conversation"; }

    async function startWith(contactToken) {
        $("side-error").textContent = "";
        try {
            const r = await api(API, "POST", { contact: contactToken });
            closeContacts(); $("conv-search").value = "";
            await loadList();
            openConversation(r.conversation);
        } catch (e) { $("side-error").textContent = e.message; }
    }

    // ------------------------------------------------------------ open conversation
    function appendMessages(msgs) {
        const box = $("msgs");
        if (!msgs.length) return;
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
        const empty = box.querySelector(".state"); if (empty) empty.remove();
        for (const m of msgs) {
            if (m.id <= lastId) continue;
            const div = document.createElement("div");
            div.className = "msg" + (m.mine ? " mine" : "");
            div.textContent = m.body;                         // text only: never rendered as HTML
            const t = document.createElement("time");
            t.textContent = (m.mine ? "You" : m.sender) + " · " + when(m.at);
            div.appendChild(t);
            box.appendChild(div);
            lastId = Math.max(lastId, m.id);
        }
        if (atBottom || msgs.some(m => m.mine)) box.scrollTop = box.scrollHeight;
    }

    async function poll() {
        if (!current || document.hidden) return;
        try {
            const d = await api(API + "/" + encodeURIComponent(current) + "/messages?after=" + lastId);
            appendMessages(d.messages || []);
            $("main-error").textContent = "";
        } catch (e) { $("main-error").textContent = e.message; }
    }

    async function openConversation(id, greeting) {
        current = id; lastId = 0;
        clearInterval(openTimer);
        root.classList.add("viewing");
        $("msgs").innerHTML = '<div class="state">Loading messages…</div>';
        $("main-error").textContent = "";
        renderList();
        try {
            const d = await api(API + "/" + encodeURIComponent(id) + "/messages");
            const w = d.conversation.with;
            $("head-who").innerHTML = "<b>" + esc(w.name) + "</b>" + metaLine(w);
            $("msgs").innerHTML = d.messages.length ? "" : '<div class="state">No messages yet. Say hello.</div>';
            appendMessages(d.messages);
            $("composer").hidden = !d.conversation.can_send;
            if (!d.conversation.can_send) $("main-error").textContent = "This account is deactivated and cannot receive messages.";
            if (greeting && !d.messages.length && !$("body").value) $("body").value = greeting;
            const c = conversations.find(x => x.id === id); if (c) { c.unread = 0; renderList(); }
            if (window.SafeDriveUnread) window.SafeDriveUnread.refresh();
            history.replaceState(null, "", location.pathname + "?c=" + encodeURIComponent(id));
            openTimer = setInterval(poll, OPEN_MS);
        } catch (e) {
            $("msgs").innerHTML = '<div class="state">' + esc(e.message) + "</div>";
            $("composer").hidden = true;
        }
    }

    async function send(ev) {
        ev.preventDefault();
        const text = $("body").value.trim();
        if (!text || !current || sending) return;
        sending = true; $("send-btn").disabled = true; $("main-error").textContent = "";
        try {
            const m = await api(API + "/" + encodeURIComponent(current) + "/messages", "POST", { body: text });
            await poll();                                    // picks up anything that arrived meanwhile
            appendMessages([m]);
            $("body").value = "";
            loadList();
        } catch (e) { $("main-error").textContent = e.message; }
        finally { sending = false; $("send-btn").disabled = false; $("body").focus(); }
    }

    // ------------------------------------------------------------ wiring
    $("conv-list").addEventListener("click", ev => { const b = ev.target.closest("button[data-id]"); if (b) openConversation(b.dataset.id); });
    $("contact-list").addEventListener("click", ev => { const b = ev.target.closest("button[data-contact]"); if (b) startWith(b.dataset.contact); });
    $("new-btn").addEventListener("click", () => {
        if ($("contact-list").hidden) { $("new-btn").textContent = "Cancel"; searchContacts(); } else closeContacts();
    });
    $("conv-search").addEventListener("input", () => {
        if (!$("contact-list").hidden) { clearTimeout(searchTimer); searchTimer = setTimeout(searchContacts, 300); }
        else renderList();
    });
    $("composer").addEventListener("submit", send);
    $("body").addEventListener("keydown", ev => { if (ev.key === "Enter" && !ev.shiftKey) send(ev); });
    $("back-btn").addEventListener("click", () => { root.classList.remove("viewing"); current = null; clearInterval(openTimer); renderList(); });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) { loadList(); poll(); } });

    (async () => {
        await loadList();
        const p = new URLSearchParams(location.search);
        try {
            if (p.get("contact_manager")) {
                const r = await api(MANAGER, "POST");
                await loadList();
                openConversation(r.conversation, r.greeting);
            } else if (p.get("to")) {
                await startWith(p.get("to"));
            } else if (p.get("c")) {
                openConversation(p.get("c"));
            }
        } catch (e) { $("side-error").textContent = e.message; }
        setInterval(loadList, LIST_MS);
    })();
})();
