import http.server


class Host(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.pages: dict[str, bytes] = {}
        self.redirects: dict[str, str] = {}
        self.statuses: dict[str, int] = {}
        self.served: list[str] = []
        self.authorizations: dict[str, str | None] = {}

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(http.server.BaseHTTPRequestHandler):
    server: Host

    def do_GET(self):
        self.server.served.append(self.path)
        self.server.authorizations[self.path] = self.headers.get("Authorization")
        if self.path in self.server.statuses:
            self.send_error(self.server.statuses[self.path])
            return
        if self.path in self.server.redirects:
            self.send_response(302)
            self.send_header("Location", self.server.redirects[self.path])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = self.server.pages.get(self.path)
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object):
        pass
