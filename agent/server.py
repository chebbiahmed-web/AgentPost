"""Web bridge between agentpost_web_dashboard.html and the agent.

The dashboard is a static page with no dataset of its own: it talks to this
server, which imports agent.py. That keeps one source of truth for the emails,
the injection classifier and the approval gate.

Run:
    python3 agent/server.py
Then open http://127.0.0.1:8765

HOST and PORT environment variables are honoured (portals set $PORT); use
HOST=0.0.0.0 to serve the dashboard to other devices on your network.
AGENTPOST_DATA_DIR points emails_db.json/outbox.json at a mounted volume.
AGENTPOST_USER (default "agentpost") and AGENTPOST_PASSWORD protect every
route with HTTP Basic auth; leave the password empty to disable auth.

Endpoints
    GET    /                 the dashboard
    GET    /api/messages     every email plus agent.classifier_email verdict
    GET    /api/outbox       replies a human approved (outbox.json)
    POST   /api/reply        {"id", "contenu", "approved"} -> traiter_message
    DELETE /api/outbox       clear outbox.json
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import agent

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "agentpost_web_dashboard.html"

# Basic auth is active only when a password is configured, so local runs stay
# frictionless while a public deployment is locked behind a login.
AUTH_USER = os.environ.get("AGENTPOST_USER", "agentpost")
AUTH_PASSWORD = os.environ.get("AGENTPOST_PASSWORD", "")


def snapshot_messages() -> list[dict[str, Any]]:
    """Every email, with the agent's own verdict attached for the UI."""
    messages = []
    for summary in agent.lister_messages():
        email = agent.lire_message(summary["id"])
        email["verdict"] = agent.classifier_email(email)
        messages.append(email)
    return messages


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentPostBridge/1.0"

    # -- helpers ---------------------------------------------------------

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def _authorized(self) -> bool:
        """True when auth is disabled or the request carries valid credentials."""
        if not AUTH_PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "basic" or not token:
            return False
        try:
            decoded = base64.b64decode(token.strip(), validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            return False
        user, sep, password = decoded.partition(":")
        if not sep:
            return False
        return hmac.compare_digest(
            user.encode("utf-8"), AUTH_USER.encode("utf-8")
        ) and hmac.compare_digest(
            password.encode("utf-8"), AUTH_PASSWORD.encode("utf-8")
        )

    def _require_auth(self) -> bool:
        """Send a 401 challenge and return False when the request is anonymous."""
        if self._authorized():
            return True
        body = b"Authentication required.\n"
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="AgentPost", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
        return False

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    # -- routing ---------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 (http.server API)
        if not self._require_auth():
            return
        self._send(HTTPStatus.NO_CONTENT, b"", "text/plain")

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        try:
            if path in ("/", "/index.html", "/agentpost_web_dashboard.html"):
                self._send(
                    HTTPStatus.OK,
                    DASHBOARD_PATH.read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/api/messages":
                self._json(HTTPStatus.OK, {"messages": snapshot_messages()})
            elif path == "/api/outbox":
                self._json(HTTPStatus.OK, {"outbox": agent.lire_outbox()})
            else:
                self._error(HTTPStatus.NOT_FOUND, f"No route for {path}")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path != "/api/reply":
            self._error(HTTPStatus.NOT_FOUND, f"No route for {path}")
            return

        try:
            payload = self._read_json()
            email_id = str(payload.get("id") or "").strip()
            contenu = str(payload.get("contenu") or "").strip()
            approved = payload.get("approved") is True

            if not email_id:
                self._error(HTTPStatus.BAD_REQUEST, "id is required")
                return
            if not contenu:
                self._error(HTTPStatus.BAD_REQUEST, "contenu cannot be empty")
                return

            if not approved:
                self._json(
                    HTTPStatus.OK,
                    {
                        "email_id": email_id,
                        "status": "rejected_by_human",
                        "sent": False,
                    },
                )
                return

            # The browser already collected the human decision; traiter_message
            # still re-classifies and refuses traps before approving anything.
            result = agent.traiter_message(
                email_id, contenu=contenu, approve=lambda: True
            )
            self._json(HTTPStatus.OK, result)
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except json.JSONDecodeError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        path = self.path.split("?", 1)[0]
        if path != "/api/outbox":
            self._error(HTTPStatus.NOT_FOUND, f"No route for {path}")
            return
        try:
            agent.OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
            agent.OUTBOX_PATH.write_text("[]\n", encoding="utf-8")
            self._json(HTTPStatus.OK, {"outbox": []})
        except Exception as exc:  # pragma: no cover - defensive
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write(f"[agentpost] {self.address_string()} {format % args}\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the AgentPost dashboard")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "8765"))
    )
    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AgentPost dashboard: http://{args.host}:{args.port}", flush=True)
    print(
        f"Emails: {agent.EMAILS_DB_PATH} | Outbox: {agent.OUTBOX_PATH}",
        flush=True,
    )
    if AUTH_PASSWORD:
        print(f"Auth: enabled (user: {AUTH_USER})", flush=True)
    else:
        print(
            "Auth: DISABLED — set AGENTPOST_PASSWORD before exposing publicly.",
            flush=True,
        )
    print("Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
