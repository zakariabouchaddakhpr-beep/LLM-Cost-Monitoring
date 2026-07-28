import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.infrastructure.persistence import TraceRepository


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        repo = TraceRepository()
        try:
            summary = repo.get_summary()
            payload = [
                {
                    "metric": "total_calls",
                    "value": summary["total_calls"],
                },
                {
                    "metric": "total_cost_usd",
                    "value": summary["total_cost_usd"],
                },
                {
                    "metric": "avg_latency_ms",
                    "value": summary["avg_latency_ms"],
                },
                {
                    "metric": "avg_quality_score",
                    "value": summary["avg_quality_score"],
                },
                {
                    "metric": "total_input_tokens",
                    "value": summary["total_input_tokens"],
                },
                {
                    "metric": "total_output_tokens",
                    "value": summary["total_output_tokens"],
                },
            ]
        finally:
            repo.close()

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def serve_forever(host="0.0.0.0", port=8001):
    server = ThreadingHTTPServer((host, port), MetricsHandler)
    server.serve_forever()
