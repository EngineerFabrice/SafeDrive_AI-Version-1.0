"""Umusare availability ON/OFF switch, and SMTP email delivery (fake SMTP server: tests never send real email)."""
import io
import json
import logging
import re
import smtplib

import pytest

from tests.web.conftest import login
from tests.web.journey_helpers import Journey, START, north
from tests.web.test_email_terms import FORM, clock, coop, outbox, register  # noqa: F401 (fixtures)

pytestmark = pytest.mark.db

AVAIL = "/assistance/umusare/availability"


def post(client, url, body=None):
    return client.post(url, data=json.dumps(body or {}), content_type="application/json")


# ================================================================== Umusare availability
class Crew:
    """One cooperative, a driver and an Umusare (status configurable), each with a logged-in client."""

    def __init__(self, app, db, status="VERIFIED"):
        self.app, self.db = app, db
        self.coop = db.cooperative("Coop A", "CA")
        self.driver = db.user("driver@example.com", coop_id=self.coop)
        self.umusare = db.user("umusare@example.com", role="umusare", coop_id=self.coop)
        db.query("UPDATE users SET phone='+250788123456' WHERE id=%s", (self.umusare,))
        db.query("INSERT INTO umusare_profiles (user_id, verification_status) VALUES (%s,%s)", (self.umusare, status))
        self.d, self.u = app.test_client(), app.test_client()
        login(self.d, "driver@example.com")
        login(self.u, "umusare@example.com")

    def on(self, pos=None):
        lat, lon = pos or north(1.0)
        return post(self.u, AVAIL, {"available": True, "lat": lat, "lon": lon, "accuracy": 10})

    def off(self):
        return post(self.u, AVAIL, {"available": False})

    def availability(self, uid=None):
        return self.db.query("SELECT availability FROM umusare_profiles WHERE user_id=%s", (uid or self.umusare,))[0][
            "availability"]

    def request(self):
        r = post(self.d, "/assistance/requests", {"lat": START[0], "lon": START[1]})
        assert r.status_code == 201, r.get_json()
        return r.get_json()


@pytest.fixture
def crew(app, db):
    return Crew(app, db)


def test_umusare_switches_on(crew):
    r = crew.on()
    assert r.status_code == 200 and r.get_json() == {"availability": "AVAILABLE"}
    assert crew.availability() == "AVAILABLE"
    assert crew.db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (crew.umusare,))[0]["n"] == 1


def test_umusare_switches_off(crew):
    crew.on()
    r = crew.off()
    assert r.status_code == 200 and r.get_json() == {"availability": "OFFLINE"}
    assert crew.availability() == "OFFLINE"
    assert crew.db.query("SELECT COUNT(*) AS n FROM user_locations WHERE user_id=%s", (crew.umusare,))[0]["n"] == 0


def test_switching_on_requires_a_location(crew):
    assert post(crew.u, AVAIL, {"available": True}).status_code == 400
    assert crew.availability() == "OFFLINE"


def test_current_state_is_returned_and_displayed(crew):
    html = crew.u.get("/umusare-dashboard").get_data(as_text=True)
    assert 'role="switch"' in html and 'aria-checked="false"' in html and "NOT AVAILABLE" in html
    assert "You are not available to receive assistance requests." in html
    crew.on()
    assert crew.u.get("/assistance/umusare/status").get_json()["profile"]["availability"] == "AVAILABLE"
    html = crew.u.get("/umusare-dashboard").get_data(as_text=True)
    assert 'aria-checked="true"' in html and ">AVAILABLE<" in html
    assert "You are available to receive assistance requests." in html
    assert "Go offline" not in html                                       # the old two-button UI is gone


def test_driver_cannot_change_umusare_availability(crew):
    crew.on()
    assert post(crew.d, AVAIL, {"available": False, "umusare_id": crew.umusare}).status_code == 403
    assert crew.availability() == "AVAILABLE"


@pytest.mark.parametrize("role", ["manager", "admin"])
def test_other_roles_and_anonymous_cannot_change_availability(crew, role):
    crew.on()
    crew.db.user(f"{role}@example.com", role=role, coop_id=crew.coop if role != "admin" else None)
    c = crew.app.test_client()
    login(c, f"{role}@example.com")
    assert post(c, AVAIL, {"available": False}).status_code == 403
    assert post(crew.app.test_client(), AVAIL, {"available": False}).status_code in (302, 401)
    assert crew.availability() == "AVAILABLE"


def test_an_umusare_cannot_change_another_umusares_availability(crew):
    other = crew.db.user("other@example.com", role="umusare", coop_id=crew.coop)
    crew.db.query("INSERT INTO umusare_profiles (user_id, verification_status, availability) VALUES (%s,'VERIFIED','AVAILABLE')",
                  (other,))
    post(crew.u, AVAIL, {"available": False, "umusare_id": other, "user_id": other})
    assert crew.availability(other) == "AVAILABLE"                        # only the signed-in Umusare's own row


@pytest.mark.parametrize("status", ["PENDING", "SUSPENDED", "REJECTED"])
def test_unverified_or_suspended_umusare_cannot_switch_on(app, db, status):
    crew = Crew(app, db, status=status)
    r = crew.on()
    assert r.status_code == 403 and crew.availability() == "OFFLINE"
    html = crew.u.get("/umusare-dashboard").get_data(as_text=True)
    assert re.search(r'id="avail-switch"[^>]*disabled', html, re.S)
    assert "Availability can be switched on after your cooperative verifies your account." in html
    crew.request()
    assert db.query("SELECT COUNT(*) AS n FROM assistance_offers")[0]["n"] == 0     # never offered a request


def test_deactivated_umusare_cannot_switch_on(crew):
    crew.db.query("UPDATE users SET is_active=0 WHERE id=%s", (crew.umusare,))
    assert crew.on().status_code in (302, 401)                            # the session no longer authenticates
    assert crew.availability() == "OFFLINE"


def test_switched_off_umusare_is_excluded_from_matching(crew):
    crew.on()
    crew.off()
    r = crew.request()
    assert r["status"] == "NO_UMUSARE_AVAILABLE"
    assert crew.db.query("SELECT COUNT(*) AS n FROM assistance_offers WHERE umusare_id=%s", (crew.umusare,))[0]["n"] == 0


def test_switched_on_eligible_umusare_is_offered_requests(crew):
    crew.on()
    r = crew.request()
    assert r["status"] == "MATCHING"
    offers = crew.db.query("SELECT status FROM assistance_offers WHERE umusare_id=%s", (crew.umusare,))
    assert [o["status"] for o in offers] == ["OFFERED"]


def test_switching_off_never_cancels_an_accepted_assistance(app, db):
    j = Journey(app, db)
    j.request()
    j.accept()
    r = post(j.u, AVAIL, {"available": False})
    assert r.status_code == 409 and "Finish" in r.get_json()["error"]
    row = db.query("SELECT status FROM assistance_requests WHERE id=%s", (j.id,))[0]
    assert row["status"] == "ACCEPTED"
    assert db.query("SELECT availability FROM umusare_profiles WHERE user_id=%s", (j.umusare_id,))[0]["availability"] == "BUSY"
    html = j.u.get("/umusare-dashboard").get_data(as_text=True)
    assert re.search(r'id="avail-switch"[^>]*disabled', html, re.S) and "ASSISTING" in html


# ================================================================== SMTP delivery (fake server)
class FakeSMTP:
    """Stands in for smtplib.SMTP: records the conversation and never touches the network."""
    instances = []
    fail_on = None

    def __init__(self, host, port, timeout=None, **kwargs):
        self.host, self.port, self.timeout = host, port, timeout
        self.calls, self.sent, self.closed = [], [], False
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def _step(self, name):
        self.calls.append(name)
        if FakeSMTP.fail_on == name:
            if name == "login":
                raise smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted")
            raise smtplib.SMTPServerDisconnected("connection lost")

    def starttls(self, context=None):
        self._step("starttls")

    def login(self, user, password):
        self._step("login")
        self.login_user = user

    def send_message(self, msg):
        self._step("send")
        self.sent.append(msg)


FAKE_PASSWORD = "fake-app-password-for-tests"


@pytest.fixture
def smtp(monkeypatch):
    from website import mailer
    FakeSMTP.instances, FakeSMTP.fail_on = [], None
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    for name, value in {"MAIL_BACKEND": "smtp", "MAIL_HOST": "smtp.gmail.com", "MAIL_PORT": "587",
                        "MAIL_USERNAME": "sender@example.com", "MAIL_PASSWORD": FAKE_PASSWORD,
                        "MAIL_USE_TLS": "true", "MAIL_FROM": "sender@example.com"}.items():
        monkeypatch.setenv(name, value)
    return FakeSMTP


def _plain(msg):
    return msg.get_body(preferencelist=("plain",)).get_content()


def test_smtp_success_sends_professional_otp_email(client, db, coop, smtp, caplog):
    caplog.set_level(logging.DEBUG)
    register(client, coop)
    [conn] = smtp.instances
    assert (conn.host, conn.port, conn.timeout) == ("smtp.gmail.com", 587, 15)
    assert conn.calls == ["starttls", "login", "send"] and conn.login_user == "sender@example.com" and conn.closed
    msg = conn.sent[0]
    assert msg["To"] == "aline@example.com" and msg["From"] == "sender@example.com"
    assert msg["Subject"] == "SafeDrive AI — Email Verification"
    text = _plain(msg)
    code = re.search(r"code is: (\d{6})", text).group(1)
    for phrase in ("SafeDrive AI", "Email Verification", "expires in 10 minutes", "Do not share this code with anyone"):
        assert phrase in text, phrase
    assert code in msg.get_body(preferencelist=("html",)).get_content()
    page = client.get("/verify-email").get_data(as_text=True)
    assert "We sent a 6-digit verification code" in page and code not in page
    otp = db.query("SELECT code_hash FROM email_otps")[0]["code_hash"]
    assert code not in otp
    assert FAKE_PASSWORD not in caplog.text and code not in caplog.text
    client.post("/verify-email", data={"code": code})                   # the emailed code works
    assert db.query("SELECT email_verified_at FROM users WHERE email='aline@example.com'")[0]["email_verified_at"]


def test_smtp_failure_is_reported_honestly_without_details(client, db, coop, smtp, caplog):
    smtp.fail_on = "login"
    r = register(client, coop)
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "We could not send the verification email" in page
    assert "We sent a 6-digit verification code" not in page
    assert "535" not in page and "SMTPAuthentication" not in page and "Username and Password" not in page
    assert smtp.instances[0].closed
    assert "SMTPAuthenticationError" in caplog.text and FAKE_PASSWORD not in caplog.text
    row = db.query("SELECT invalidated_at FROM email_otps")[0]
    assert row["invalidated_at"] is not None                             # the undelivered code can never be used


def test_failed_starttls_still_closes_the_connection(client, db, coop, smtp):
    smtp.fail_on = "starttls"
    register(client, coop)
    assert smtp.instances[0].closed and smtp.instances[0].calls == ["starttls"]


def test_resend_after_smtp_failure_respects_cooldown_and_hourly_limit(client, db, coop, smtp, clock):
    smtp.fail_on = "send"
    register(client, coop)
    for _ in range(5):
        r = client.post("/verify-email/resend", follow_redirects=True)
    assert len(smtp.instances) == 1                                        # cooldown: no new SMTP connection
    assert "A new verification code was sent" not in r.get_data(as_text=True)
    for _ in range(10):
        db.query("UPDATE email_otps SET created_at = created_at - INTERVAL 61 SECOND")
        clock["offset"] += 61
        client.post("/verify-email/resend")
    assert len(smtp.instances) == 5                                        # hourly cap: 5 attempts in total


def test_resend_success_message_only_after_real_handoff(client, db, coop, smtp, clock):
    smtp.fail_on = "send"
    register(client, coop)
    db.query("UPDATE email_otps SET created_at = created_at - INTERVAL 61 SECOND")
    clock["offset"] += 61
    smtp.fail_on = None
    r = client.post("/verify-email/resend", follow_redirects=True)
    page = r.get_data(as_text=True)
    assert "A new verification code was sent" in page and "We sent a 6-digit verification code" in page


def test_smtp_failure_does_not_reveal_whether_an_email_exists(app, db, coop, smtp):
    smtp.fail_on = "send"
    db.user("taken@example.com", coop_id=coop)
    new, dup = app.test_client(), app.test_client()
    register(new, coop, email="fresh@example.com")
    register(dup, coop, email="taken@example.com")
    pages = [re.sub(r"[a-z0-9]{1,4}\*\*\*@example\.com|name=\"csrf_token\" value=\"[^\"]+\"|data-wait=\"\d+\"", "X",
                    c.get("/verify-email").get_data(as_text=True)) for c in (new, dup)]
    assert pages[0] == pages[1] and "We could not send the verification email" in pages[0]


# ================================================================== configuration / fallback
@pytest.mark.parametrize("env, settings, expected", [
    ("development", {"MAIL_HOST": "smtp.gmail.com", "MAIL_USERNAME": "a@example.com", "MAIL_PASSWORD": "x"}, "smtp"),
    ("development", {"MAIL_HOST": "smtp.gmail.com"}, "file"),              # .env.example copied, no credentials yet
    ("development", {}, "file"),
    ("testing", {"MAIL_HOST": "smtp.gmail.com", "MAIL_USERNAME": "a@example.com", "MAIL_PASSWORD": "x"}, "memory"),
    ("production", {"MAIL_HOST": "smtp.gmail.com", "MAIL_USERNAME": "a@example.com", "MAIL_PASSWORD": "x"}, "smtp"),
    ("production", {}, None),
    ("production", {"MAIL_BACKEND": "file"}, None),
])
def test_transport_selection(monkeypatch, env, settings, expected):
    from website import mailer
    for name in ("MAIL_BACKEND", "MAIL_HOST", "MAIL_USERNAME", "MAIL_PASSWORD"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("SAFEDRIVE_ENV", env)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    assert mailer.backend() == expected


def test_tests_never_use_a_real_smtp_account():
    import os
    from website import mailer
    assert os.environ["MAIL_BACKEND"] == "memory" and os.environ["MAIL_PASSWORD"] == ""
    assert mailer.backend() == "memory"


def test_development_without_smtp_writes_to_the_outbox(monkeypatch, tmp_path):
    from website import mailer
    for name in ("MAIL_BACKEND", "MAIL_HOST", "MAIL_USERNAME", "MAIL_PASSWORD"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    monkeypatch.setattr(mailer, "DEV_OUTBOX_DIR", str(tmp_path))
    mailer.send("someone@example.com", "Subject", "Body")
    files = list(tmp_path.glob("*.eml"))
    assert len(files) == 1 and b"Body" in files[0].read_bytes()


def test_production_without_smtp_refuses_to_send(monkeypatch):
    from website import mailer
    for name in ("MAIL_BACKEND", "MAIL_HOST", "MAIL_USERNAME", "MAIL_PASSWORD"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    with pytest.raises(mailer.MailError):
        mailer.send("someone@example.com", "Subject", "Body")


# ================================================================== development SMTP check (CLI)
def test_smtp_test_cli_sends_without_printing_secrets(smtp, monkeypatch):
    from website import mailer
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    out = io.StringIO()
    assert mailer.smtp_test("me@example.com", out=out) == 0
    text = out.getvalue()
    assert "OK" in text and FAKE_PASSWORD not in text and "MAIL_PASSWORD: set" in text
    msg = smtp.instances[0].sent[0]
    assert msg["Subject"] == "SafeDrive AI: SMTP test" and not re.search(r"\b\d{6}\b", _plain(msg))   # no code


def test_smtp_test_cli_reports_failure(smtp, monkeypatch):
    from website import mailer
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    smtp.fail_on = "login"
    out = io.StringIO()
    assert mailer.smtp_test("me@example.com", out=out) == 1 and FAKE_PASSWORD not in out.getvalue()
    assert "FAILED at AUTH" in out.getvalue() and "App Password" in out.getvalue()


def test_username_that_is_not_an_email_is_flagged(smtp, monkeypatch, caplog):
    """Root cause seen in practice: MAIL_USERNAME set to the App Password's *name* instead of the Gmail address."""
    from website import mailer
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    monkeypatch.setenv("MAIL_USERNAME", "SafeDriveAI")
    smtp.fail_on = "login"
    out = io.StringIO()
    assert mailer.smtp_test("me@example.com", out=out) == 1
    assert "MAIL_USERNAME is not an email address" in out.getvalue()
    assert "Gmail SMTP requires the full Gmail address" in caplog.text and FAKE_PASSWORD not in caplog.text


@pytest.mark.parametrize("fail_on, stage", [("starttls", "STARTTLS"), ("login", "AUTH"), ("send", "SEND")])
def test_smtp_failure_log_names_the_stage_without_secrets(smtp, monkeypatch, caplog, fail_on, stage):
    from website import mailer
    smtp.fail_on = fail_on
    with pytest.raises(mailer.MailError) as err:
        mailer.send("someone@example.com", "Subject", "Body 123456")
    assert err.value.stage.upper() == stage
    assert f"Email delivery failed at {stage}" in caplog.text
    assert FAKE_PASSWORD not in caplog.text and "123456" not in caplog.text


def test_smtp_test_cli_refuses_in_production_and_without_smtp(monkeypatch):
    from website import mailer
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    assert mailer.smtp_test("me@example.com", out=io.StringIO()) == 2
    monkeypatch.setenv("SAFEDRIVE_ENV", "development")
    for name in ("MAIL_BACKEND", "MAIL_HOST", "MAIL_USERNAME", "MAIL_PASSWORD"):
        monkeypatch.setenv(name, "")
    out = io.StringIO()
    assert mailer.smtp_test("me@example.com", out=out) == 2 and "Nothing was sent" in out.getvalue()
