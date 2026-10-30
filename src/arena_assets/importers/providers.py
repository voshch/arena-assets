"""Host providers: stdlib urllib access to the upstream file stores importers fetch from."""

from __future__ import annotations

import abc
import http.client
import json
import logging
import os
import re
import time
import typing
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from email.message import Message
from pathlib import Path

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
TRANSPORT_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException)

_T = typing.TypeVar("_T")


def transient(error: Exception) -> bool:
    """Whether another attempt can succeed: the host answered with a retryable status, or did not answer."""
    return error.code in _TRANSIENT_STATUS if isinstance(error, urllib.error.HTTPError) else True


class Provider(abc.ABC):
    name: typing.ClassVar[str]
    timeout: typing.ClassVar[float] = 30.0
    attempts: typing.ClassVar[int] = 4
    backoff: typing.ClassVar[float] = 1.0

    def _request(self, url: str, **kwargs: typing.Any) -> urllib.request.Request:
        return urllib.request.Request(url, **kwargs)

    def _open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _retry(self, url: str, call: Callable[[], _T]) -> _T:
        """Result of *call*, tried up to `attempts` times with a doubling pause while the failure is transient."""
        for attempt in range(1, self.attempts):
            try:
                return call()
            except TRANSPORT_ERRORS as error:
                if not transient(error):
                    raise
                pause = self.backoff * 2 ** (attempt - 1)
                logger.warning("%s failed (%s), attempt %d of %d in %g s", url, error, attempt + 1, self.attempts, pause)
                time.sleep(pause)
        return call()

    @abc.abstractmethod
    def url(self, relpath: str) -> str: ...

    def read(self, url: str) -> tuple[bytes, Message]:
        """Body and headers of an absolute *url* on this host."""

        def call() -> tuple[bytes, Message]:
            with self._open(self._request(url)) as resp:
                return resp.read(), resp.headers

        return self._retry(url, call)

    def open(self, relpath: str) -> http.client.HTTPResponse:
        """Streamed body of *relpath*, for the caller to read and close."""
        url = self.url(relpath)
        return self._retry(url, lambda: self._open(self._request(url)))

    def size(self, relpath: str) -> int:
        url = self.url(relpath)

        def call() -> int:
            with self._open(self._request(url, method="HEAD")) as resp:
                return int(resp.headers["Content-Length"])

        return self._retry(url, call)

    def get(self, relpath: str, dest: Path, *, range: tuple[int, int] | None = None) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if range is None and dest.is_file():
            return dest
        url = self.url(relpath)
        part = dest.with_name(dest.name + ".part")

        def call() -> None:
            req = self._request(url)
            if range is not None:
                req.add_header("Range", f"bytes={range[0]}-{range[1]}")
            written = 0
            with self._open(req) as resp, open(part, "wb") as f:
                while chunk := resp.read(_CHUNK):
                    written += f.write(chunk)
                announced = resp.headers.get("Content-Length")
            if announced is not None and written != int(announced):
                raise ConnectionError(f"body ended after {written} of {announced} bytes")

        self._retry(url, call)
        part.replace(dest)
        return dest

    def list(self, prefix: str = "") -> Iterable[str]:
        raise NotImplementedError(f"{self.name} has no listing")


class Http(Provider):
    name = "http"

    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def url(self, relpath: str) -> str:
        return f"{self.base}/{urllib.parse.quote(relpath.lstrip('/'))}"


def hf_token() -> str | None:
    """Token in the Hugging Face lookup order: HF_TOKEN, then HF_TOKEN_PATH, then $HF_HOME/token."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    home = os.environ.get("HF_HOME") or os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "huggingface")
    path = Path(os.environ.get("HF_TOKEN_PATH") or os.path.join(home, "token"))
    return (path.read_text().strip() or None) if path.is_file() else None


class HFHub(Provider):
    name = "hfhub"
    _HOST = "https://huggingface.co"

    def __init__(self, repo: str, repo_type: typing.Literal["dataset", "model"] = "dataset", revision: str = "main", token: str | None = None, host: str = _HOST):
        self.repo = repo
        self.host = host
        self.repo_type = repo_type
        self.revision = revision
        self.token = token if token is not None else hf_token()

    def _request(self, url: str, **kwargs: typing.Any) -> urllib.request.Request:
        req = super()._request(url, **kwargs)
        if self.token:
            req.add_unredirected_header("Authorization", f"Bearer {self.token}")
        return req

    @property
    def _prefix(self) -> str:
        return f"{self.host}/datasets/{self.repo}" if self.repo_type == "dataset" else f"{self.host}/{self.repo}"

    def url(self, relpath: str) -> str:
        return f"{self._prefix}/resolve/{urllib.parse.quote(self.revision, safe='')}/{urllib.parse.quote(relpath.lstrip('/'))}"

    def _tree_url(self, prefix: str, recursive: bool = True) -> str:
        path = f"/{urllib.parse.quote(prefix.strip('/'))}" if prefix.strip("/") else ""
        return f"{self.host}/api/{self.repo_type}s/{self.repo}/tree/{urllib.parse.quote(self.revision, safe='')}{path}?recursive={str(recursive).lower()}"

    def _entries(self, prefix: str, recursive: bool) -> Iterator[dict]:
        url: str | None = self._tree_url(prefix, recursive)
        while url:
            body, headers = self.read(url)
            yield from json.loads(body)
            match = _LINK_NEXT.search(headers.get("Link", ""))
            url = match.group(1) if match else None

    def list(self, prefix: str = "") -> Iterator[str]:
        return (entry["path"] for entry in self._entries(prefix, recursive=True) if entry["type"] == "file")

    def directories(self, prefix: str = "") -> Iterator[str]:
        """Paths of the directories directly under prefix."""
        return (entry["path"] for entry in self._entries(prefix, recursive=False) if entry["type"] == "directory")


class Fuel(Provider):
    name = "fuel"
    _HOST = "https://fuel.gazebosim.org"
    _API = "1.0"
    _PER_PAGE = 100

    def __init__(self, owner: str):
        self.owner = owner

    @property
    def _models(self) -> str:
        return f"{self._HOST}/{self._API}/{urllib.parse.quote(self.owner, safe='')}/models"

    def url(self, relpath: str) -> str:
        return f"{self._models}/{urllib.parse.quote(relpath.lstrip('/'))}"

    def _page_url(self, page: int) -> str:
        return f"{self._models}?{urllib.parse.urlencode({'page': page, 'per_page': self._PER_PAGE})}"

    def page(self, page: int = 1) -> tuple[list[dict], str | None]:
        return self._page(self._page_url(page))

    def _page(self, url: str) -> tuple[list[dict], str | None]:
        body, headers = self.read(url)
        link = headers.get("Link", "")
        entries = json.loads(body)
        match = _LINK_NEXT.search(link)
        return entries, urllib.parse.urljoin(self._HOST, match.group(1)) if match else None

    def models(self) -> Iterator[dict]:
        url: str | None = self._page_url(1)
        while url:
            entries, url = self._page(url)
            yield from entries

    def list(self, prefix: str = "") -> Iterator[str]:
        for model in self.models():
            if model["name"].startswith(prefix):
                yield model["name"]


class PolyHaven(Provider):
    name = "polyhaven"
    _API = "https://api.polyhaven.com"
    _FILES = "https://dl.polyhaven.org"
    _AGENT = "arena-assets"

    def __init__(self, api: str = _API, files: str = _FILES):
        self.api = api.rstrip("/")
        self.files_host = files.rstrip("/")

    def _request(self, url: str, **kwargs: typing.Any) -> urllib.request.Request:
        req = super()._request(url, **kwargs)
        req.add_header("User-Agent", self._AGENT)
        return req

    def url(self, relpath: str) -> str:
        """API URL of a relative path, a download URL of the file host as it is."""
        return relpath if relpath.startswith(f"{self.files_host}/") else f"{self.api}/{urllib.parse.quote(relpath.lstrip('/'))}"

    def assets(self, kind: str = "textures") -> dict[str, dict]:
        return json.loads(self.read(f"{self.api}/assets?{urllib.parse.urlencode({'t': kind})}")[0])

    def list(self, prefix: str = "") -> Iterator[str]:
        return (name for name in sorted(self.assets()) if name.startswith(prefix))


class AmbientCG(Provider):
    name = "ambientcg"
    _HOST = "https://ambientcg.com"
    _PER_PAGE = 500
    _AGENT = "arena-assets"

    def __init__(self, host: str = _HOST):
        self.host = host.rstrip("/")

    def _request(self, url: str, **kwargs: typing.Any) -> urllib.request.Request:
        req = super()._request(url, **kwargs)
        req.add_header("User-Agent", self._AGENT)
        return req

    def url(self, relpath: str) -> str:
        return f"{self.host}/get?{urllib.parse.urlencode({'file': relpath})}"

    def _query(self, **query: object) -> list[dict]:
        return json.loads(self.read(f"{self.host}/api/v2/full_json?{urllib.parse.urlencode(query)}")[0])["foundAssets"]

    def asset(self, asset: str) -> dict | None:
        """API record of one asset with its downloads, None when the host has no asset of that id."""
        return next(iter(self._query(id=asset, include="downloadData,tagData,displayData")), None)

    def assets(self, kind: str = "Material") -> Iterator[dict]:
        offset = 0
        while page := self._query(type=kind, sort="Alphabet", limit=self._PER_PAGE, offset=offset):
            yield from page
            offset += len(page)

    def list(self, prefix: str = "") -> Iterator[str]:
        return (asset["assetId"] for asset in self.assets() if asset["assetId"].startswith(prefix))
