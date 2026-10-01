# website/history.py
"""Technical monitoring history (Phase 8): sessions, statistics and events.

Design:
- ``HistoryRecorder`` runs in its own thread. It samples the engine's
  snapshot once per second and never runs inside the camera/processing loop.
- Nothing is written per frame. Statistics are kept in memory and saved
  together with any new events in one small transaction every few seconds.
- Database errors are logged (rate-limited) and retried later. Events wait in
  a bounded in-memory buffer; monitoring itself is never affected.
- Only technical facts are stored. Model events record *that* a model ran or
  failed, never its predicted class or probabilities, so mock output cannot
  become "impairment history".
"""
import collections
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from engine.features import FEATURE_SCHEMA_VERSION

from . import get_connection

log = logging.getLogger(__name__)

SAMPLE_INTERVAL = 1.0        # seconds between snapshot samples
FLUSH_INTERVAL = 5.0         # seconds between database writes
EVENT_COOLDOWN = 60.0        # min seconds between two events of the same type in a session
FEATURE_STREAK = 3           # consecutive samples before FEATURES_UNAVAILABLE is recorded
MAX_PENDING_EVENTS = 1000    # in-memory buffer while the database is unavailable
DB_CONNECT_TIMEOUT = 3       # seconds; keeps history requests from hanging when MySQL is down
STALE_SESSION_SECONDS = 120  # RUNNING sessions not updated for this long are marked INTERRUPTED
MOCK_LABEL = "MOCK / DEVELOPMENT_ONLY"

MODEL_FAILURE_STATUSES = ("MODEL_UNAVAILABLE", "MODEL_ERROR", "SCHEMA_MISMATCH", "INVALID_INPUT")


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class HistoryUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- storage
class HistoryStore:
    """Plain-SQL access to the monitoring history tables (same style as the rest of the app)."""

    SESSION_COLUMNS = ("id", "started_by", "started_at", "ended_at", "status", "frames_processed",
                       "average_processing_fps", "average_latency_ms", "model_provider", "model_name",
                       "model_version", "model_is_mock", "model_development_only", "feature_schema_version")
    EVENT_COLUMNS = ("session_id", "timestamp", "frame_id", "event_type", "status", "model_provider",
                     "model_name", "model_version", "feature_schema_version", "is_mock",
                     "development_only", "error_message")

    def __init__(self, connect=None):
        self._connect = connect or (lambda: get_connection(connect_timeout=DB_CONNECT_TIMEOUT))

    def _transaction(self, work):
        try:
            conn = self._connect()
        except Exception as exc:
            raise HistoryUnavailable(type(exc).__name__) from exc
        try:
            cursor = conn.cursor()
            cursor.execute("SET time_zone = '+00:00'")   # DB defaults (created_at/updated_at) in UTC too
            result = work(cursor)
            conn.commit()
            return result
        except Exception as exc:
            conn.rollback()
            raise HistoryUnavailable(type(exc).__name__) from exc
        finally:
            conn.close()

    def write(self, sessions, events):
        """Upsert sessions, then insert events, in one transaction."""
        cols = self.SESSION_COLUMNS
        upsert = (f"INSERT INTO monitoring_sessions ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
                  "ON DUPLICATE KEY UPDATE ended_at=VALUES(ended_at), status=VALUES(status), "
                  "frames_processed=VALUES(frames_processed), "
                  "average_processing_fps=VALUES(average_processing_fps), "
                  "average_latency_ms=VALUES(average_latency_ms)")
        insert = (f"INSERT INTO monitoring_events ({', '.join(self.EVENT_COLUMNS)}) "
                  f"VALUES ({', '.join(['%s'] * len(self.EVENT_COLUMNS))})")

        def work(cursor):
            for s in sessions:
                cursor.execute(upsert, tuple(s[c] for c in cols))
            if events:
                cursor.executemany(insert, [tuple(e[c] for c in self.EVENT_COLUMNS) for e in events])
        self._transaction(work)

    def mark_interrupted(self, exclude_id=None):
        """Close sessions left RUNNING by a crash or kill; returns how many were marked."""
        def work(cursor):
            return cursor.execute(
                # keep updated_at: it is the last time the session was known to be running
                "UPDATE monitoring_sessions SET status='INTERRUPTED', updated_at=updated_at WHERE status='RUNNING' "
                "AND updated_at < UTC_TIMESTAMP(3) - INTERVAL %s SECOND AND id <> %s",
                (STALE_SESSION_SECONDS, exclude_id or ""))
        return self._transaction(work)

    def list_sessions(self, limit=20):
        def work(cursor):
            cursor.execute("SELECT s.*, (SELECT COUNT(*) FROM monitoring_events e WHERE e.session_id = s.id) "
                           "AS event_count FROM monitoring_sessions s ORDER BY started_at DESC LIMIT %s", (limit,))
            return cursor.fetchall()
        return self._transaction(work)

    def get_session(self, session_id, event_limit=500):
        def work(cursor):
            cursor.execute("SELECT * FROM monitoring_sessions WHERE id=%s", (session_id,))
            session = cursor.fetchone()
            if session is None:
                return None, []
            cursor.execute(f"SELECT {', '.join(('id',) + self.EVENT_COLUMNS)} FROM monitoring_events "
                           "WHERE session_id=%s ORDER BY timestamp, id LIMIT %s", (session_id, event_limit))
            return session, cursor.fetchall()
        return self._transaction(work)


# ---------------------------------------------------------------- recording
class _Session:
    def __init__(self, started_by, model_info):
        self.id = str(uuid.uuid4())
        self.started_by = started_by
        self.started_at = utcnow()
        self.ended_at = None
        self.status = "RUNNING"
        self.model_provider = model_info.provider if model_info else "none"
        self.model_name = model_info.name if model_info else None
        self.model_version = model_info.version if model_info else None
        self.model_is_mock = bool(model_info and model_info.is_mock)
        self.model_development_only = bool(model_info and model_info.development_only)
        # statistics
        self.frames_processed = 0
        self.first_frame_at = None          # monotonic time when processing began
        self.last_sample_at = None
        self.latency_sum = 0.0
        self.latency_count = 0
        # event tracking
        self.camera_down = False
        self.detectors_reported = False
        self.feature_streak = 0
        self.last_event = {}

    @property
    def average_fps(self):
        if not self.frames_processed or self.first_frame_at is None or self.last_sample_at is None:
            return None
        elapsed = self.last_sample_at - self.first_frame_at
        return round(self.frames_processed / elapsed, 2) if elapsed > 0 else None

    @property
    def average_latency(self):
        return round(self.latency_sum / self.latency_count, 1) if self.latency_count else None

    def row(self):
        return {
            "id": self.id, "started_by": self.started_by, "started_at": self.started_at,
            "ended_at": self.ended_at, "status": self.status, "frames_processed": self.frames_processed,
            "average_processing_fps": self.average_fps, "average_latency_ms": self.average_latency,
            "model_provider": self.model_provider, "model_name": self.model_name,
            "model_version": self.model_version, "model_is_mock": int(self.model_is_mock),
            "model_development_only": int(self.model_development_only),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }


class HistoryRecorder:
    def __init__(self, engine, store=None):
        self.engine = engine
        self.store = store or HistoryStore()
        self._lock = threading.Lock()
        self._session = None                 # active session
        self._unsaved_sessions = {}          # id -> _Session with changes not yet written
        self._events = collections.deque()
        self._event_seq = 0
        self._wake = threading.Event()
        self._thread = None
        self._last_flush = 0.0
        self._last_error_log = 0.0
        self._recovered = False
        self.db_ok = None
        self.last_error = ""
        self.dropped_events = 0

    # ------------------------------------------------------------ lifecycle (request threads)
    def start_session(self, started_by=None):
        model = self.engine.impairment_model
        with self._lock:
            if self._session is not None:
                self._close_locked("INTERRUPTED")
            session = _Session(started_by, model.info if model else None)
            self._session = session
            self._unsaved_sessions[session.id] = session
            self._event_locked(session, "SESSION_STARTED", status="RUNNING")
            if model is not None and not model.available:
                self._event_locked(session, "MODEL_UNAVAILABLE", status="MODEL_UNAVAILABLE",
                                   error=model.unavailable_reason)
        self._ensure_thread()
        self._wake.set()
        return session.id

    def end_session(self, status="COMPLETED"):
        with self._lock:
            if self._session is None:
                return None
            self._sample_locked(self._session)   # final statistics (the snapshot keeps them after stop)
            session_id = self._session.id
            self._close_locked(status)
        self._wake.set()
        return session_id

    def shutdown(self, timeout=3.0):
        """On process exit: close an open session as INTERRUPTED and try one last write."""
        self.end_session("INTERRUPTED")
        deadline = time.monotonic() + timeout
        while self._has_pending() and time.monotonic() < deadline:
            if not self._flush():
                break

    def status(self):
        with self._lock:
            return {"database_ok": self.db_ok, "last_error": self.last_error,
                    "pending_events": len(self._events), "dropped_events": self.dropped_events,
                    "active_session_id": self._session.id if self._session else None}

    # ------------------------------------------------------------ internals (lock held)
    def _close_locked(self, status):
        s = self._session
        s.status, s.ended_at = status, utcnow()
        self._event_locked(s, "SESSION_ENDED", status=status)
        self._unsaved_sessions[s.id] = s
        self._session = None

    def _event_locked(self, s, event_type, status=None, error=None, frame_id=None, impairment=None,
                      cooldown=False):
        now = time.monotonic()
        if cooldown and now - s.last_event.get(event_type, -1e9) < EVENT_COOLDOWN:
            return
        s.last_event[event_type] = now
        if len(self._events) >= MAX_PENDING_EVENTS:
            self._events.popleft()
            self.dropped_events += 1
        self._event_seq += 1
        self._events.append({
            "seq": self._event_seq,
            "session_id": s.id, "timestamp": utcnow(), "frame_id": frame_id, "event_type": event_type,
            "status": status, "model_provider": s.model_provider, "model_name": s.model_name,
            "model_version": s.model_version, "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "is_mock": int(impairment["is_mock"] if impairment else s.model_is_mock),
            "development_only": int(impairment["development_only"] if impairment else s.model_development_only),
            "error_message": (error or None) and str(error)[:500],
        })

    def _sample_locked(self, s):
        snap = self.engine.snapshot()
        now = time.monotonic()
        s.last_sample_at = now
        if snap.frames_processed and s.first_frame_at is None:
            s.first_frame_at = now
        s.frames_processed = max(s.frames_processed, snap.frames_processed)
        if snap.frame_latency is not None and snap.status.value != "STOPPED":
            s.latency_sum += snap.frame_latency
            s.latency_count += 1
        self._unsaved_sessions[s.id] = s
        if snap.status.value == "STOPPED":
            return

        camera = snap.camera_status.value
        if camera == "UNAVAILABLE" and not s.camera_down:
            s.camera_down = True
            self._event_locked(s, "CAMERA_UNAVAILABLE", status=camera, error=snap.message)
        elif camera == "CONNECTED" and s.camera_down:
            s.camera_down = False
            self._event_locked(s, "CAMERA_RECONNECTED", status=camera, frame_id=snap.frame_id)

        if snap.model_status.value == "UNAVAILABLE" and not s.detectors_reported:
            s.detectors_reported = True
            self._event_locked(s, "DETECTION_MODEL_UNAVAILABLE", status=snap.model_status.value,
                               error=snap.message)

        s.feature_streak = s.feature_streak + 1 if snap.status.value == "FEATURES_UNAVAILABLE" else 0
        if s.feature_streak == FEATURE_STREAK:
            reason = snap.features.get("reason") if snap.features else snap.message
            self._event_locked(s, "FEATURES_UNAVAILABLE", status=snap.features_status.value,
                               error=reason, frame_id=snap.frame_id, cooldown=True)

        imp = snap.impairment
        if imp:
            if imp["status"] in MODEL_FAILURE_STATUSES:
                self._event_locked(s, imp["status"], status=imp["status"], error=imp.get("error"),
                                   frame_id=imp.get("frame_id"), impairment=imp, cooldown=True)
            elif imp["valid"]:
                # Record only that the model produced output. The predicted class and
                # probabilities are deliberately not stored.
                error = MOCK_LABEL if imp["is_mock"] else None
                self._event_locked(s, "MODEL_OUTPUT", status=imp["status"], error=error,
                                   frame_id=imp.get("frame_id"), impairment=imp, cooldown=True)

    # ------------------------------------------------------------ background thread
    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="safedrive-history", daemon=True)
            self._thread.start()

    def _has_pending(self):
        with self._lock:
            return bool(self._events or self._unsaved_sessions)

    def _run(self):
        while True:
            self._wake.wait(SAMPLE_INTERVAL)
            self._wake.clear()
            try:
                with self._lock:
                    s = self._session
                    if s is not None:
                        if not self.engine.is_running:
                            self._sample_locked(s)
                            self._close_locked("INTERRUPTED")
                        else:
                            self._sample_locked(s)
                    closing = s is None and bool(self._unsaved_sessions)
                if closing or time.monotonic() - self._last_flush >= FLUSH_INTERVAL:
                    self._flush()
            except Exception:
                log.exception("History recorder iteration failed")

    def _flush(self):
        """Write unsaved sessions and queued events; returns True on success."""
        with self._lock:
            sessions = list(self._unsaved_sessions.values())
            rows = [s.row() for s in sessions]
            events = list(self._events)
            active_id = self._session.id if self._session else None
        if not rows and not events:
            return True
        self._last_flush = time.monotonic()
        try:
            if not self._recovered:
                marked = self.store.mark_interrupted(exclude_id=active_id)
                if marked:
                    log.info("Marked %d stale monitoring session(s) as INTERRUPTED", marked)
                self._recovered = True
            self.store.write(rows, events)
        except HistoryUnavailable as exc:
            with self._lock:
                self.db_ok, self.last_error = False, f"history database unavailable ({exc})"
            if time.monotonic() - self._last_error_log > 60:
                log.warning("Monitoring history not saved: %s; will retry", exc)
                self._last_error_log = time.monotonic()
            return False
        with self._lock:
            written = events[-1]["seq"] if events else 0
            while self._events and self._events[0]["seq"] <= written:   # newer events stay queued
                self._events.popleft()
            for s in sessions:
                if s is not self._session and self._unsaved_sessions.get(s.id) is s:
                    del self._unsaved_sessions[s.id]
            if self._session is not None:
                self._unsaved_sessions[self._session.id] = self._session
            self.db_ok, self.last_error = True, ""
        return True


# ---------------------------------------------------------------- API serialisation
def _iso(value):
    return value.isoformat(timespec="milliseconds") + "Z" if isinstance(value, datetime) else value


def session_to_dict(row):
    data = {k: _iso(v) for k, v in row.items()}
    for key in ("model_is_mock", "model_development_only"):
        data[key] = bool(row.get(key))
    end = row.get("ended_at") or (row.get("updated_at") if row.get("status") != "RUNNING" else None)
    start = row.get("started_at")
    data["duration_seconds"] = round((end - start).total_seconds(), 1) if end and start else None
    data["label"] = MOCK_LABEL if data["model_development_only"] else None
    return data


def event_to_dict(row):
    data = {k: _iso(v) for k, v in row.items()}
    for key in ("is_mock", "development_only"):
        data[key] = bool(row.get(key))
    return data
