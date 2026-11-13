import configparser
import contextlib
import gzip
import hashlib
import http.client
import json
import os
import sqlite3
import subprocess
import threading
import time
import typing
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.logging import format_file_size, get_logger, get_manager

DEFAULT_URL = "https://storage.arena-rosnav.org"
DEFAULT_SOURCE = "public/models"
BUCKETS = ("public", "arena-internal")
MAX_UPLOAD = 4 << 30
TOKEN_HINT = "set ARENA_ASSETS_TOKEN or log in with 'gcloud auth login'"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"


class StoreError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _reason(e: urllib.error.HTTPError) -> str:
    body = e.read().decode(errors="replace").strip()
    with contextlib.suppress(ValueError, TypeError, KeyError):
        body = json.loads(body)["error"]
    return str(body or e.reason)


def store_url() -> str:
    """Base URL of the asset store, `$ARENA_ASSETS_URL` or the public host."""
    return (os.environ.get("ARENA_ASSETS_URL") or DEFAULT_URL).rstrip("/")


def assets_root() -> Path | None:
    """Download cache, `$ARENA_ASSETS_DIR` or `$ARENA_DATA_DIR/assets`, None with neither set."""
    if root := os.environ.get("ARENA_ASSETS_DIR"):
        return Path(root)
    if data := os.environ.get("ARENA_DATA_DIR"):
        return Path(data) / "assets"
    return None


def cache_root() -> Path:
    root = assets_root()
    if root is None:
        raise StoreError("no asset cache directory: set ARENA_DATA_DIR or ARENA_ASSETS_DIR")
    return root


def index_ttl() -> int:
    return int(os.environ.get("ARENA_MODELS_TTL", "86400"))


def _gcloud_stored_token() -> str:
    """Access token refreshed from the active gcloud account's credentials in `$CLOUDSDK_CONFIG`."""
    config = Path(os.environ.get("CLOUDSDK_CONFIG") or Path.home() / ".config" / "gcloud")
    parser = configparser.ConfigParser()
    parser.read(config / "configurations" / f"config_{(config / 'active_config').read_text().strip()}")
    with contextlib.closing(sqlite3.connect(f"{(config / 'credentials.db').as_uri()}?mode=ro", uri=True)) as db:
        (value,) = db.execute("SELECT value FROM credentials WHERE account_id = ?", (parser["core"]["account"],)).fetchone()
    credentials = json.loads(value)
    body = urllib.parse.urlencode({"grant_type": "refresh_token", **{key: credentials[key] for key in ("client_id", "client_secret", "refresh_token")}}).encode()
    with urllib.request.urlopen(credentials.get("token_uri") or GOOGLE_TOKEN_URI, body) as resp:
        return json.load(resp)["access_token"]


def resolve_token() -> str:
    """Bearer token from `$ARENA_ASSETS_TOKEN`, else the gcloud CLI, else gcloud's stored credentials."""
    if token := os.environ.get("ARENA_ASSETS_TOKEN"):
        return token
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        return subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return _gcloud_stored_token()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, configparser.Error) as e:
        raise StoreError(f"no asset store token: {TOKEN_HINT}") from e


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def outermost(assets: typing.Iterable[str]) -> list[str]:
    """Asset directories that lie inside no other listed asset directory."""
    listed = set(assets)
    return [asset for asset in dict.fromkeys(assets) if not any(parent.as_posix() in listed for parent in PurePosixPath(asset).parents)]


def _under(prefix: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/" if prefix else ""


class Bucket:
    """One source on the asset store, `<bucket>[/<folder>]`, answering listings from the bucket index."""

    def __init__(self, source: str, token: str | None = None):
        self.name = source
        self.token = token
        self._base = store_url()
        self.bucket, _, folder = source.strip("/").partition("/")
        self._folder = _under(folder)
        self._index: dict[str, dict] | None = None
        self._revalidated = False
        self._lock = threading.Lock()
        self._token_lock = threading.Lock()

    def _url(self, name: str) -> str:
        return f"{self._base}/{self.bucket}/{urllib.parse.quote(name, safe='/')}"

    def _object_url(self, name: str) -> str:
        return self._url(self._folder + name)

    @staticmethod
    def _request(url: str, token: str | None = None, **kwargs: typing.Any) -> urllib.request.Request:
        req = urllib.request.Request(url, **kwargs)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        return req

    def _token(self) -> str:
        with self._token_lock:
            if self.token is None:
                self.token = resolve_token()
            return self.token

    def _urlopen(self, url: str, token: str | None, **kwargs: typing.Any) -> http.client.HTTPResponse:
        try:
            return urllib.request.urlopen(self._request(url, token, **kwargs))
        except urllib.error.URLError as e:
            if isinstance(e, urllib.error.HTTPError):
                raise
            raise StoreError(f"cannot reach the asset store at {self._base}: {e.reason}") from None

    def _what(self, url: str, method: str) -> str:
        return f"{method} {self.bucket}/{urllib.parse.unquote(url.removeprefix(f'{self._base}/{self.bucket}/'))}"

    def _refused(self, e: urllib.error.HTTPError, what: str, token: str | None) -> StoreError:
        reason = _reason(e)
        if e.code == 401:
            hint = TOKEN_HINT if token is None else "the token was not accepted, it may have expired"
        elif e.code == 403:
            hint = f"{self._base}/_api/whoami shows what the token may read and write"
        elif e.code == 412:
            hint = "it exists with other content and --no-overwrite keeps it"
        else:
            hint = ""
        return StoreError(f"{what}: {e.code} {reason}" + (f", {hint}" if hint else ""), status=e.code)

    def _open(self, url: str, **kwargs: typing.Any) -> http.client.HTTPResponse:
        """Open *url* anonymously until the store refuses, then with a resolved token."""
        what = self._what(url, kwargs.get("method", "GET"))
        token = self.token
        try:
            return self._urlopen(url, token, **kwargs)
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403):
                raise
            if token is not None:
                raise self._refused(e, what, token) from None
            try:
                token = self._token()
            except StoreError:
                raise self._refused(e, what, None) from None
        try:
            return self._urlopen(url, token, **kwargs)
        except urllib.error.HTTPError as e:
            if e.code not in (401, 403):
                raise
            raise self._refused(e, what, token) from None

    def _send(self, url: str, **kwargs: typing.Any) -> http.client.HTTPResponse:
        """A write: authenticated from the start, a refusal raised with the store's reason."""
        token = self._token()
        try:
            return self._urlopen(url, token, **kwargs)
        except urllib.error.HTTPError as e:
            raise self._refused(e, self._what(url, kwargs["method"]), token) from None

    @property
    def _index_path(self) -> Path:
        return cache_root() / ".index" / f"{self.bucket}.json"

    def _fetch_index(self, revalidate: bool) -> bytes:
        """The bucket index from disk while younger than ARENA_MODELS_TTL, else revalidated by ETag."""
        path = self._index_path
        etag_path = path.with_suffix(".etag")
        try:
            cached = path.read_bytes()
            age = time.time() - path.stat().st_mtime
        except OSError:
            cached, age = None, 0.0
        if cached is not None and not revalidate and age < index_ttl():
            return cached
        headers = {"Accept-Encoding": "gzip"}
        if cached is not None and etag_path.is_file():
            headers["If-None-Match"] = etag_path.read_text().strip()
        self._revalidated = True
        try:
            with self._open(self._url("index.json"), headers=headers) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                etag = resp.headers.get("ETag")
        except urllib.error.HTTPError as e:
            if e.code == 304 and cached is not None:
                with contextlib.suppress(OSError):
                    os.utime(path)
                return cached
            if e.code == 404:
                raise StoreError(f"no bucket {self.bucket!r} on {self._base}, known buckets are {', '.join(BUCKETS)}", status=404) from None
            raise StoreError(f"GET {self.bucket}/index.json: {e.code} {_reason(e)}", status=e.code) from None
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(body)
            os.replace(tmp, path)
            if etag:
                etag_path.write_text(etag)
            else:
                etag_path.unlink(missing_ok=True)
        return body

    def _load(self, revalidate: bool = False) -> dict[str, dict]:
        objects = json.loads(self._fetch_index(revalidate))["objects"]
        return {entry["name"].removeprefix(self._folder): entry for entry in objects if entry["name"].startswith(self._folder)}

    def objects(self) -> dict[str, dict]:
        """Index entries of this source by object name, relative to its folder."""
        with self._lock:
            if self._index is None:
                self._index = self._load()
            return self._index

    def _refresh(self) -> bool:
        """Revalidate the index over the network, at most once per instance."""
        with self._lock:
            if self._revalidated:
                return False
            self._index = self._load(revalidate=True)
            return True

    def _stale(self) -> None:
        """A write changed the bucket: the next process revalidates the index."""
        with contextlib.suppress(OSError):
            os.utime(self._index_path, (0, 0))

    def _matching(self, keep: typing.Callable[[str], bool]) -> list[dict]:
        found = [entry for name, entry in self.objects().items() if keep(name)]
        if not found and self._refresh():
            found = [entry for name, entry in self.objects().items() if keep(name)]
        return found

    def listdir(self, prefix: str) -> list[dict]:
        """Index entries under a prefix, named relative to the source."""
        prefix = _under(prefix)
        return [{**entry, "name": entry["name"].removeprefix(self._folder)} for entry in self._matching(lambda name: name.startswith(prefix))]

    def prefix_exists(self, prefix: str) -> bool:
        """Whether any object lives under *prefix*."""
        return bool(self.listdir(prefix))

    def list_assets(self, prefix: str = "") -> list[str]:
        """Asset directories under a prefix: the parents of annotation files."""
        prefix = _under(prefix)
        names = [entry["name"].removeprefix(self._folder) for entry in self._matching(lambda name: name.startswith(prefix) and PurePosixPath(name).name == ANNOTATION_NAME)]
        return outermost([os.path.dirname(name) for name in names])

    def listchildren(self, prefix: str, containing: str | None = None) -> list[str]:
        """Immediate child directories under *prefix*. With *containing* only children holding a file of that name at any depth count."""
        prefix = _under(prefix)
        relative = [entry["name"].removeprefix(self._folder).removeprefix(prefix) for entry in self._matching(lambda name: name.startswith(prefix))]
        if containing:
            relative = [name for name in relative if PurePosixPath(name).name == containing]
        return [f"{prefix}{child}" for child in dict.fromkeys(name.split("/")[0] for name in relative if "/" in name)]

    def object_exists(self, name: str) -> bool:
        """Existence of an exact object, with no sentinel assumptions."""
        if name in self.objects():
            return True
        return self._refresh() and name in self.objects()

    def listdirs(self, prefixes: list[str]) -> dict[str, list[dict]]:
        """Index entries under several prefixes."""
        return {prefix: self.listdir(prefix) for prefix in prefixes}

    def asset_exists(self, asset: str, sentinel: str | None = ANNOTATION_NAME) -> bool:
        """Whether `<asset>/<sentinel>` exists, or with *sentinel* None any object at or under *asset*."""
        if sentinel is None:
            return self.object_exists(asset) or self.prefix_exists(asset)
        if os.path.basename(asset) != sentinel:
            asset = os.path.join(asset, sentinel)
        return self.object_exists(asset)

    def assets_exist(self, assets: list[str], sentinel: str | None = ANNOTATION_NAME) -> list[tuple[str, bool]]:
        return [(asset, self.asset_exists(asset, sentinel)) for asset in assets]

    def _receive(self, name: str, part: str) -> int:
        """Append the rest of *name* to *part*, from its current size on."""
        offset = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            resp = self._open(self._object_url(name), headers=headers)
        except urllib.error.HTTPError as e:
            if e.code == 416 and offset:
                return 0
            raise StoreError(f"GET {self.name}/{name}: {e.code} {_reason(e)}", status=e.code) from None
        size = 0
        with resp, open(part, "ab" if resp.status == 206 else "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                size += len(chunk)
        return size

    def download(self, name: str, local_path: str) -> int:
        """Stream one object to a local file, resuming a partial one and checking the index sha256. Returns bytes received."""
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        part = f"{local_path}.part"
        received = 0
        expected = actual = None
        for _ in range(2):
            received += self._receive(name, part)
            expected = self.objects().get(name, {}).get("sha256")
            actual = sha256_of(part)
            if expected is None or actual == expected:
                os.replace(part, local_path)
                return received
            os.remove(part)
            self._refresh()
        raise StoreError(f"{name}: downloaded sha256 {actual} does not match the index's {expected}")

    def upload(self, local_path: str, name: str, sha256: str | None = None, overwrite: bool = True) -> int:
        """Stream a local file to an object. Returns bytes sent."""
        size = os.path.getsize(local_path)
        if size > MAX_UPLOAD:
            raise StoreError(f"{local_path} is {size} bytes, the store takes at most {MAX_UPLOAD}")
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "X-Content-SHA256": sha256 or sha256_of(local_path),
        }
        if not overwrite:
            headers["If-None-Match"] = "*"
        with open(local_path, "rb") as f:
            self._send(self._object_url(name), data=f, method="PUT", headers=headers).close()
        self._stale()
        return size

    def delete(self, name: str) -> None:
        """Remove one object, one already gone counts as removed. The store allows it to its uploader or an admin."""
        try:
            self._send(self._object_url(name), method="DELETE").close()
        except StoreError as e:
            if e.status != 404:
                raise
        self._stale()

    def under(self, prefix: str) -> list[str]:
        """Names of the object *prefix* and of every object below it."""
        prefix = prefix.strip("/")
        if not prefix:
            raise StoreError("refusing to select a whole source, name a path inside it")
        return sorted(name for name in self.objects() if name == prefix or name.startswith(f"{prefix}/"))

    def browse(self, path: str = "") -> tuple[dict[str, tuple[int, int]], list[dict]]:
        """Folders (object count and bytes below each) and files directly under *path*."""
        prefix = _under(path)
        folders: dict[str, tuple[int, int]] = {}
        files: list[dict] = []
        for entry in self.listdir(prefix):
            rest = entry["name"].removeprefix(prefix)
            folder, slash, _ = rest.partition("/")
            if slash:
                count, size = folders.get(folder, (0, 0))
                folders[folder] = (count + 1, size + int(entry.get("size", 0)))
            else:
                files.append({**entry, "name": rest})
        return dict(sorted(folders.items())), sorted(files, key=lambda entry: entry["name"])


def fetch_database(
    bucket: str,
    bucket_paths: list[str],
    destination: str,
    annotations: bool = True,
    model_formats: list[str] | None = None,
) -> None:
    """Download files from multiple paths of an asset store source.

    Args:
        bucket: The source to download from, `<bucket>[/<folder>]`
        bucket_paths: List of paths inside the source to download
        destination: The local directory path to download to
        annotations: Whether to download annotation files (default: True)
        model_formats: List of model formats to download (e.g., ['usdz']). If None, download all formats
    """
    logger = get_logger("fetch")
    try:
        b = Bucket(bucket)
        os.makedirs(destination, exist_ok=True)

        logger.info("Scanning %d path(s) for files...", len(bucket_paths))
        all_blobs_by_path = b.listdirs(bucket_paths)

        seen = set()
        file_blobs = []
        for path, blobs in all_blobs_by_path.items():
            for blob in blobs:
                name = blob["name"]
                if name in seen or name.endswith("/"):
                    continue
                if not annotations and os.path.basename(name) == ANNOTATION_NAME:
                    continue
                if model_formats is not None:
                    if not any(any(part.endswith(f".{fmt.lower()}") for part in name.lower().split("/")) for fmt in model_formats):
                        continue
                seen.add(name)
                file_blobs.append((path, blob))

        total_files = len(file_blobs)
        total_size = sum(int(blob.get("size", 0)) for _, blob in file_blobs)

        if total_files == 0:
            logger.warning("No files found for the given paths in bucket '%s'", bucket)
            return

        logger.info("Found %d files (%s) to download", total_files, format_file_size(total_size))

        downloaded_count = 0
        downloaded_size = 0
        skipped_size = 0

        with get_manager() as manager:
            status_bar = manager.status_bar(
                status_format="Downloading: {current_file}{fill}",
                current_file="Initializing...",
                color="cyan",
            )

            data_progress = manager.counter(
                total=total_size,
                desc=f"[{1:>{len(str(total_files))}d}/{total_files}]",
                color="blue",
                counter_format="{desc}{desc_pad}{count_formatted}/{total_formatted} [{elapsed}, {rate_formatted}/s]{fill}",
                bar_format="{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count_formatted}/{total_formatted} [{elapsed}<{eta}, {rate_formatted}/s]",
                fields={
                    "count_formatted": format_file_size(0),
                    "total_formatted": format_file_size(total_size),
                    "rate_formatted": format_file_size(0),
                },
            )

            pending = []

            for path, blob in file_blobs:
                blob_name = blob["name"]
                blob_size = int(blob.get("size", 0))

                prefix = path.strip("/")
                if prefix:
                    prefix += "/"
                relative_blob_path = blob_name[len(prefix) :] if prefix and blob_name.startswith(prefix) else blob_name
                local_file_path = os.path.join(destination, prefix, relative_blob_path.lstrip("/"))

                if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == blob_size and sha256_of(local_file_path) == blob.get("sha256"):
                    logger.info(
                        "Skipping '%s' - already exists with matching sha256",
                        blob_name,
                    )
                    downloaded_count += 1
                    skipped_size += blob_size

                    new_total = total_size - skipped_size
                    data_progress.total = new_total

                    elapsed_time = data_progress.elapsed
                    current_rate = downloaded_size / elapsed_time if elapsed_time > 0 else 0

                    data_progress.desc = f"[{downloaded_count:>{len(str(total_files))}d}/{total_files}]"
                    data_progress.update(
                        0,
                        count_formatted=format_file_size(downloaded_size),
                        total_formatted=format_file_size(new_total),
                        rate_formatted=format_file_size(current_rate),
                    )
                    continue

                pending.append((blob_name, blob_size, local_file_path))

            with ThreadPoolExecutor(max_workers=20) as pool:
                futures = {pool.submit(b.download, blob_name, local_file_path): (blob_name, blob_size) for blob_name, blob_size, local_file_path in pending}
                try:
                    for future in as_completed(futures):
                        blob_name, blob_size = futures[future]
                        try:
                            future.result()
                        except StoreError as e:
                            if e.status != 404:
                                raise
                            logger.warning("Skipping '%s' - listed in the index but gone from the store", blob_name)
                            continue

                        downloaded_count += 1
                        downloaded_size += blob_size

                        status_bar.update(current_file=f"{blob_name} ({format_file_size(blob_size)})")

                        elapsed_time = data_progress.elapsed
                        current_rate = downloaded_size / elapsed_time if elapsed_time > 0 else 0

                        data_progress.desc = f"[{downloaded_count:>{len(str(total_files))}d}/{total_files}]"
                        data_progress.update(
                            blob_size,
                            count_formatted=format_file_size(downloaded_size),
                            total_formatted=format_file_size(total_size - skipped_size),
                            rate_formatted=format_file_size(current_rate),
                        )
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

            status_bar.update(current_file="Complete!")

        logger.info(
            "Downloaded %d files (%s) successfully to: %s",
            downloaded_count,
            format_file_size(downloaded_size),
            destination,
        )

    except Exception as e:
        logger.error("Fetch error: %s", e)
        raise
