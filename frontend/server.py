from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=str(root), **kwargs)  # noqa: E731
    server = ThreadingHTTPServer(("0.0.0.0", 8080), handler)
    print("Frontend running at http://0.0.0.0:8080")
    server.serve_forever()


if __name__ == "__main__":
    main()

