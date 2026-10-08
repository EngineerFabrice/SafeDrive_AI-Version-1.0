# website/chat.py
"""Internal chat page and JSON API (logic and authorization live in chat_service).

Every endpoint requires login; POSTs are CSRF-protected (X-CSRFToken header).
Nothing the browser sends decides who may talk to whom.
"""
from flask import Blueprint, jsonify, render_template, request
from flask_login import current_user

from . import ROLES
from . import chat_service as svc
from . import notifications as notes
from .assistance_service import AssistanceError
from .auth import roles_required

chat = Blueprint("chat", __name__, url_prefix="/chat")


@chat.errorhandler(AssistanceError)
def chat_error(err):
    return jsonify({"error": err.message, "code": err.code}), err.http_status


def _uid():
    return int(current_user.get_id())


def _body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@chat.route("/")
@roles_required(*ROLES)
def page():
    return render_template("chat.html", body_max=svc.BODY_MAX)


@chat.route("/api/conversations")
@roles_required(*ROLES)
def conversations():
    return jsonify({"conversations": svc.list_conversations(_uid())})


@chat.route("/api/conversations", methods=["POST"])
@roles_required(*ROLES)
def open_conversation():
    return jsonify(svc.open_conversation(_uid(), _body().get("contact"))), 201


@chat.route("/api/contact-manager", methods=["POST"])
@roles_required(*ROLES)
def contact_manager():
    return jsonify(svc.contact_manager(_uid())), 201


@chat.route("/api/conversations/<public_id>/messages")
@roles_required(*ROLES)
def messages(public_id):
    return jsonify(svc.read_messages(_uid(), public_id, request.args.get("after")))


@chat.route("/api/conversations/<public_id>/messages", methods=["POST"])
@roles_required(*ROLES)
def send(public_id):
    return jsonify(svc.send_message(_uid(), public_id, _body().get("body"))), 201


@chat.route("/api/contacts")
@roles_required(*ROLES)
def contacts():
    return jsonify({"contacts": svc.contacts(_uid(), request.args.get("q", ""))})


@chat.route("/api/unread")
@roles_required(*ROLES)
def unread():
    return jsonify(svc.unread_total(_uid()))


@chat.route("/api/notifications")
@roles_required(*ROLES)
def notifications():
    return jsonify({"notifications": notes.recent(_uid())})


@chat.route("/api/notifications/read", methods=["POST"])
@roles_required(*ROLES)
def notifications_read():
    notes.mark_all_read(_uid())
    return jsonify({"ok": True})
