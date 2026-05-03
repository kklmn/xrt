# -*- coding: utf-8 -*-
"""
Small standard-library web server for the xrtGlowWeb prototype.
"""

import argparse
import json
import mimetypes
import signal
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .adapter import (
    DEFAULT_MAX_RAYS, PropagationSession, apply_scene_updates, clean_json,
    load_layout, scene_from_layout)


STATIC_DIR = Path(__file__).with_name("static")


class XrtGlowWebState:
    def __init__(self, layout, max_rays=DEFAULT_MAX_RAYS):
        self.layout = layout
        self.scene = scene_from_layout(layout)
        self.session = PropagationSession(layout, max_rays=max_rays)


def json_bytes(payload):
    return json.dumps(clean_json(payload), separators=(",", ":")).encode(
        "utf-8")


def read_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


class XrtGlowWebHandler(BaseHTTPRequestHandler):
    server_version = "xrtGlowWeb/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (
            self.client_address[0], self.log_date_time_string(), fmt % args))

    @property
    def state(self):
        return self.server.state

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/layout":
                self.send_json(self.state.layout)
            elif parsed.path == "/api/scene":
                self.send_json(self.state.scene)
            elif parsed.path == "/api/status":
                self.send_json(self.state.session.status())
            elif parsed.path in ("", "/"):
                self.send_static("index.html")
            else:
                self.send_static(parsed.path.lstrip("/"))
        except Exception as exc:
            self.send_error_json(exc)

    def do_POST(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/run":
                max_rays = int(params.get("maxRays", [DEFAULT_MAX_RAYS])[0])
                timeout = float(params.get("timeout", [20])[0])
                messages = self.state.session.run_once(
                    timeout=timeout, max_rays=max_rays)
                updates = self.state.session.scene_updates()
                self.state.scene = apply_scene_updates(
                    self.state.scene, updates)
                self.send_json({"messages": messages,
                                "scene": self.state.scene,
                                "status": self.state.session.status()})
            elif parsed.path == "/api/modify":
                payload = read_json(self)
                self.state.session.modify(
                    payload.get("uuid"),
                    payload.get("object_type", "oe"),
                    payload.get("kwargs", {}))
                self.send_json({"ok": True,
                                "status": self.state.session.status()})
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as exc:
            self.send_error_json(exc)

    def send_json(self, payload, status=200):
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, exc):
        self.send_json({
            "error": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }, status=500)

    def send_static(self, name):
        if "/" in name:
            parts = [part for part in name.split("/") if part]
            path = STATIC_DIR.joinpath(*parts)
        else:
            path = STATIC_DIR / name
        path = path.resolve()
        static_root = STATIC_DIR.resolve()
        if not str(path).startswith(str(static_root)) or not path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        body = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class XrtGlowWebServer(ThreadingHTTPServer):
    def __init__(self, address, state):
        super().__init__(address, XrtGlowWebHandler)
        self.state = state


def serve(host="127.0.0.1", port=8765, layout=None, max_rays=DEFAULT_MAX_RAYS):
    state = XrtGlowWebState(layout, max_rays=max_rays)
    server = XrtGlowWebServer((host, int(port)), state)

    def shutdown(*_args):
        state.session.stop()
        server.shutdown()

    try:
        signal.signal(signal.SIGTERM, shutdown)
    except Exception:
        pass

    print("xrtGlowWeb listening on http://{}:{}/".format(host, port))
    try:
        server.serve_forever()
    finally:
        state.session.stop()
        server.server_close()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Serve an experimental browser xrtGlow viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--layout", help="Path to an xrtQook/xrtGlow JSON or XML layout.")
    parser.add_argument(
        "--beamline",
        help="Python module_or_path:object returning a BeamLine or layout dict.")
    parser.add_argument("--max-rays", type=int, default=DEFAULT_MAX_RAYS)
    parser.add_argument(
        "--print-scene", action="store_true",
        help="Print the browser scene payload and exit.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    layout = load_layout(layout_path=args.layout, beamline_spec=args.beamline)
    if args.print_scene:
        print(json.dumps(clean_json(scene_from_layout(layout)), indent=2))
        return 0
    serve(host=args.host, port=args.port, layout=layout, max_rays=args.max_rays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
