"""Internal chat: role/cooperative authorization, IDOR resistance, unread counts, CSRF (MySQL test database)."""
import json

import pytest

from tests.web.conftest import login
from tests.web.test_cooperative_verification import World

pytestmark = pytest.mark.db

API = "/chat/api/conversations"


def post(client, url, body=None):
    return client.post(url, data=json.dumps(body or {}), content_type="application/json")


class Chat(World):
    def __init__(self, app, db):
        super().__init__(app, db)
        self.clients = {}

    def c(self, email):
        if email not in self.clients:
            self.clients[email] = self.client(email)
        return self.clients[email]

    def verify(self, *uids):
        for uid in uids:
            for table in ("driver_profiles", "umusare_profiles"):
                self.db.query(f"UPDATE {table} SET verification_status='VERIFIED' WHERE user_id=%s", (uid,))

    def token(self, viewer, target):
        from website.chat_service import contact_token
        with self.app.test_request_context():
            return contact_token(viewer, target)

    def open(self, email, viewer, target):
        return post(self.c(email), API, {"contact": self.token(viewer, target)})

    def contacts(self, email, q=""):
        return {p["name"] for p in self.c(email).get(f"/chat/api/contacts?q={q}").get_json()["contacts"]}


@pytest.fixture
def w(app, db):
    return Chat(app, db)


# ---------------------------------------------------------------- contact lists are server-filtered
def test_pending_driver_can_only_find_own_manager(w):
    assert w.contacts("drv.a@example.com") == {"mgr.a"}


def test_verified_member_finds_verified_own_cooperative_members_and_manager_only(w):
    w.verify(w.drv_a, w.ums_a, w.drv_b, w.ums_b)
    assert w.contacts("drv.a@example.com") == {"mgr.a", "ums.a"}
    assert w.contacts("ums.a@example.com") == {"mgr.a", "drv.a"}


def test_manager_finds_own_cooperative_and_admins_only(w):
    assert w.contacts("mgr.a@example.com") == {"admin", "drv.a", "ums.a"}


def test_admin_can_search_everyone(w):
    assert w.contacts("admin@example.com") >= {"mgr.a", "mgr.b", "drv.a", "drv.b", "ums.a", "ums.b"}
    assert w.contacts("admin@example.com", "drv.b") == {"drv.b"}


# ---------------------------------------------------------------- allowed conversations
def test_driver_contacts_manager_and_manager_replies(w):
    r = post(w.c("drv.a@example.com"), "/chat/api/contact-manager")
    assert r.status_code == 201 and r.get_json()["greeting"].startswith("Hello, I have registered")
    conv = r.get_json()["conversation"]
    assert post(w.c("drv.a@example.com"), f"{API}/{conv}/messages",
                {"body": "Hello, I have registered and I am waiting for cooperative verification."}).status_code == 201
    note = w.db.query("SELECT kind, title FROM notifications WHERE user_id=%s", (w.mgr_a,))
    assert note[0]["kind"] == "SUPPORT_MESSAGE" and note[0]["title"] == "New verification/support message from drv.a"
    assert w.c("mgr.a@example.com").get("/chat/api/unread").get_json()["messages"] == 1
    msgs = w.c("mgr.a@example.com").get(f"{API}/{conv}/messages").get_json()
    assert msgs["conversation"]["type"] == "VERIFICATION" and msgs["messages"][0]["body"].startswith("Hello")
    assert w.c("mgr.a@example.com").get("/chat/api/unread").get_json()["messages"] == 0       # read now
    assert post(w.c("mgr.a@example.com"), f"{API}/{conv}/messages", {"body": "Please come to the office."}).status_code == 201
    assert w.c("drv.a@example.com").get("/chat/api/unread").get_json()["messages"] == 1
    listed = w.c("drv.a@example.com").get(API).get_json()["conversations"]
    assert listed[0]["unread"] == 1 and listed[0]["with"]["role"] == "manager"
    acts = [r["action"] for r in w.db.query("SELECT action FROM audit_logs")]
    assert {"CHAT_CONVERSATION_CREATED", "MANAGER_CONTACTED", "CHAT_MESSAGE_SENT"} <= set(acts)
    details = " ".join(str(r["details"]) for r in w.db.query("SELECT details FROM audit_logs"))
    assert "office" not in details and "registered and I am" not in details               # no message text in audit


def test_one_conversation_per_pair(w):
    first = w.open("mgr.a@example.com", w.mgr_a, w.drv_a).get_json()
    again = w.open("mgr.a@example.com", w.mgr_a, w.drv_a).get_json()
    assert first["created"] and not again["created"] and first["conversation"] == again["conversation"]
    assert w.db.query("SELECT COUNT(*) AS n FROM conversations")[0]["n"] == 1


@pytest.mark.parametrize("sender, receiver", [("drv.a", "ums.a"), ("ums.a", "drv.a")])
def test_verified_members_of_same_cooperative_can_chat(w, sender, receiver):
    w.verify(w.drv_a, w.ums_a)
    ids = {"drv.a": w.drv_a, "ums.a": w.ums_a}
    r = w.open(f"{sender}@example.com", ids[sender], ids[receiver])
    assert r.status_code == 201
    conv = r.get_json()["conversation"]
    assert post(w.c(f"{sender}@example.com"), f"{API}/{conv}/messages", {"body": "hi"}).status_code == 201
    assert w.c(f"{receiver}@example.com").get(f"{API}/{conv}/messages").get_json()["messages"][0]["body"] == "hi"


def test_admin_messages_anyone_and_they_can_reply_but_not_start(w):
    conv = w.open("admin@example.com", w.admin, w.drv_b).get_json()["conversation"]
    assert post(w.c("admin@example.com"), f"{API}/{conv}/messages", {"body": "Security check"}).status_code == 201
    assert post(w.c("drv.b@example.com"), f"{API}/{conv}/messages", {"body": "OK"}).status_code == 201
    assert w.open("ums.b@example.com", w.ums_b, w.admin).status_code == 404               # members cannot start with admin
    assert w.open("mgr.b@example.com", w.mgr_b, w.admin).status_code == 201               # managers can


# ---------------------------------------------------------------- forbidden conversations
@pytest.mark.parametrize("email, viewer, target", [
    ("mgr.a@example.com", "mgr_a", "drv_b"), ("mgr.a@example.com", "mgr_a", "ums_b"),
    ("mgr.a@example.com", "mgr_a", "mgr_b"), ("drv.a@example.com", "drv_a", "drv_b"),
    ("drv.a@example.com", "drv_a", "mgr_b"), ("ums.a@example.com", "ums_a", "ums_b"),
    ("ums.a@example.com", "ums_a", "drv_b"),
])
def test_cross_cooperative_conversation_is_blocked(w, email, viewer, target):
    w.verify(w.drv_a, w.drv_b, w.ums_a, w.ums_b)                                          # even when everyone is verified
    assert w.open(email, getattr(w, viewer), getattr(w, target)).status_code == 404
    assert w.db.query("SELECT COUNT(*) AS n FROM conversations")[0]["n"] == 0


def test_unverified_members_cannot_chat_with_each_other(w):
    assert w.open("drv.a@example.com", w.drv_a, w.ums_a).status_code == 404
    w.verify(w.drv_a)
    assert w.open("drv.a@example.com", w.drv_a, w.ums_a).status_code == 404               # target still unverified


def test_contact_tokens_are_bound_to_the_viewer_and_signed(w):
    stolen = w.token(w.mgr_a, w.drv_a)                                                     # issued to manager A
    assert post(w.c("drv.b@example.com"), API, {"contact": stolen}).status_code == 404
    for bad in ("", "123", str(w.drv_a), stolen[:-2] + "xx", None):
        assert post(w.c("mgr.a@example.com"), API, {"contact": bad}).status_code == 404


def test_non_participants_cannot_view_or_send(w):
    w.verify(w.ums_a)
    conv = w.open("drv.a@example.com", w.drv_a, w.mgr_a).get_json()["conversation"]
    post(w.c("drv.a@example.com"), f"{API}/{conv}/messages", {"body": "private"})
    for email in ("ums.a@example.com", "mgr.b@example.com", "drv.b@example.com", "admin@example.com"):
        c = w.c(email)
        assert c.get(f"{API}/{conv}/messages").status_code == 404, email
        assert post(c, f"{API}/{conv}/messages", {"body": "intrusion"}).status_code == 404, email
        assert all(x["id"] != conv for x in c.get(API).get_json()["conversations"])
    assert w.db.query("SELECT COUNT(*) AS n FROM messages")[0]["n"] == 1


def test_conversation_ids_are_not_guessable_or_numeric(w):
    conv = w.open("mgr.a@example.com", w.mgr_a, w.drv_a).get_json()["conversation"]
    assert len(conv) == 22 and not conv.isdigit()
    numeric = w.db.query("SELECT id FROM conversations")[0]["id"]
    for probe in (str(numeric), "../" + conv, "A" * 22):
        assert w.c("mgr.a@example.com").get(f"{API}/{probe}/messages").status_code == 404


def test_leaving_the_cooperative_closes_existing_conversations(w):
    conv = w.open("drv.a@example.com", w.drv_a, w.mgr_a).get_json()["conversation"]
    w.db.query("UPDATE cooperative_memberships SET cooperative_id=%s WHERE user_id=%s", (w.b, w.drv_a))
    assert w.c("mgr.a@example.com").get(f"{API}/{conv}/messages").status_code == 404
    assert post(w.c("drv.a@example.com"), f"{API}/{conv}/messages", {"body": "still there?"}).status_code == 404
    assert w.c("mgr.a@example.com").get(API).get_json()["conversations"] == []


def test_deactivated_user_cannot_receive_messages(w):
    conv = w.open("mgr.a@example.com", w.mgr_a, w.drv_a).get_json()["conversation"]
    w.db.query("UPDATE users SET is_active=0 WHERE id=%s", (w.drv_a,))
    r = post(w.c("mgr.a@example.com"), f"{API}/{conv}/messages", {"body": "hello"})
    assert r.status_code == 409


def test_message_validation(w):
    conv = w.open("mgr.a@example.com", w.mgr_a, w.drv_a).get_json()["conversation"]
    c = w.c("mgr.a@example.com")
    assert post(c, f"{API}/{conv}/messages", {"body": "   "}).status_code == 400
    assert post(c, f"{API}/{conv}/messages", {"body": "x" * 2001}).status_code == 400
    r = post(c, f"{API}/{conv}/messages", {"body": "<script>alert(1)</script>"})
    assert r.status_code == 201 and r.get_json()["body"] == "<script>alert(1)</script>"   # stored as text; UI renders text


def test_chat_requires_login(client, db):
    assert client.get("/chat/").status_code == 302
    assert client.get(API).status_code in (302, 401)


def test_chat_page_renders_for_every_role(w):
    for email in ("admin@example.com", "mgr.a@example.com", "drv.a@example.com", "ums.a@example.com"):
        html = w.c(email).get("/chat/").get_data(as_text=True)
        assert 'id="chat"' in html and "Messages" in html and 'name="csrf-token"' in html


def test_notifications_are_private_and_mark_read(w):
    post(w.c("drv.a@example.com"), "/chat/api/contact-manager")
    w.client("mgr.a@example.com").post(f"/manager/members/{w.drv_a}/review", data={"action": "VERIFY"})
    assert w.c("drv.a@example.com").get("/chat/api/unread").get_json()["notifications"] == 1
    own = w.c("drv.a@example.com").get("/chat/api/notifications").get_json()["notifications"]
    assert [n["kind"] for n in own] == ["VERIFICATION_APPROVED"]
    assert w.c("drv.b@example.com").get("/chat/api/notifications").get_json()["notifications"] == []
    post(w.c("drv.a@example.com"), "/chat/api/notifications/read")
    assert w.c("drv.a@example.com").get("/chat/api/unread").get_json()["notifications"] == 0


# ---------------------------------------------------------------- CSRF stays enforced
def test_chat_posts_require_csrf_token(db):
    from website import create_app
    app = create_app({"WTF_CSRF_ENABLED": True, "BCRYPT_LOG_ROUNDS": 4, "TESTING": True})
    coop = db.cooperative()
    db.user("mgr@example.com", role="manager", coop_id=coop)
    db.user("d@example.com", coop_id=coop)
    c = app.test_client()
    page = c.get("/login").get_data(as_text=True)
    csrf = page.split('name="csrf_token" value="')[1].split('"')[0]
    c.post("/login", data={"email": "d@example.com", "password": "Passw0rd!", "csrf_token": csrf})
    r = c.post("/chat/api/contact-manager", data="{}", content_type="application/json")
    assert r.status_code == 400 and "CSRF" in r.get_json()["error"]
    r = c.post("/chat/api/contact-manager", data="{}", content_type="application/json", headers={"X-CSRFToken": csrf})
    assert r.status_code == 201
    conv = r.get_json()["conversation"]
    assert c.post(f"{API}/{conv}/messages", data=json.dumps({"body": "x"}),
                  content_type="application/json").status_code == 400
    assert db.query("SELECT COUNT(*) AS n FROM messages")[0]["n"] == 0
