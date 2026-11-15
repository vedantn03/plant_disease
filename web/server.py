"""Local static server for the farmer app and evaluator dashboard routes."""

from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class LeafwiseHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/runtime-config.js":
            api_url = os.environ.get("LEAFWISE_API_URL", "").strip()
            body = f"window.LEAFWISE_API_URL = {json.dumps(api_url)};\n".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.split("?", 1)[0] in {"/insights", "/insights/"}:
            self.path = "/insights.html"
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        # Keep normal access logs, but avoid logging uploaded image contents (never sent here).
        super().log_message(format, *args)


def main() -> None:
    web_root = Path(__file__).resolve().parent
    port = int(os.environ.get("LEAFWISE_UI_PORT", "8501"))
    handler = partial(LeafwiseHandler, directory=str(web_root))
    with ThreadingHTTPServer(("0.0.0.0", port), handler) as server:
        print(f"LEAFWISE static UI listening on port {port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
