import json
import os
import typing
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from arena_models.impl import ANNOTATION_NAME
from arena_models.utils.logging import format_file_size, get_logger, get_manager


class Bucket:
    _API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    _DOWNLOAD = "https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{obj}?alt=media"
    _UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"

    def __init__(self, bucket: str, token: str | None = None):
        self.name = bucket
        self.token = token
        self._api = self._API.format(bucket=bucket)

    def _object_url(self, obj: str, **params: str) -> str:
        encoded = urllib.parse.quote(obj, safe="")
        query = urllib.parse.urlencode(params)
        if obj:
            return f"{self._api}/{encoded}?{query}" if query else f"{self._api}/{encoded}"
        return f"{self._api}?{query}" if query else self._api

    def _upload_url(self, obj: str) -> str:
        query = urllib.parse.urlencode({"uploadType": "media", "name": obj})
        return f"{self._UPLOAD.format(bucket=self.name)}?{query}"

    def _request(self, url: str, **kwargs: typing.Any) -> urllib.request.Request:
        req = urllib.request.Request(url, **kwargs)
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        return req

    def listdir(self, prefix: str) -> list[dict]:
        """List all objects under a prefix using the GCS JSON API."""
        results = []
        prefix = prefix.strip("/")
        if prefix:
            prefix += "/"

        url = self._object_url("", prefix=prefix)

        while url:
            resp = urllib.request.urlopen(self._request(url))
            data = json.loads(resp.read())
            results.extend(data.get("items", []))

            next_token = data.get("nextPageToken")
            url = self._object_url("", prefix=prefix, pageToken=next_token) if next_token else None

        return results

    def list_assets(self, prefix: str = "") -> list[str]:
        """List asset directories under a prefix: one server-side glob on the annotation file, names only."""
        prefix = prefix.strip("/")
        if prefix:
            prefix += "/"
        params = {
            "prefix": prefix,
            "matchGlob": f"**/{ANNOTATION_NAME}",
            "fields": "items(name),nextPageToken",
            "maxResults": "5000",
        }
        assets = []
        url = self._object_url("", **params)
        while url:
            data = json.loads(urllib.request.urlopen(self._request(url)).read())
            assets.extend(os.path.dirname(item["name"]) for item in data.get("items", []))
            next_token = data.get("nextPageToken")
            url = self._object_url("", **params, pageToken=next_token) if next_token else None
        return assets

    def listdirs(self, prefixes: list[str]) -> dict[str, list[dict]]:
        """List objects under multiple prefixes concurrently."""
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(self.listdir, p): p for p in prefixes}
            return {futures[f]: f.result() for f in futures}

    def asset_exists(self, asset: str) -> bool:
        """Check if an asset exists in the bucket."""
        if os.path.basename(asset) != ANNOTATION_NAME:
            asset = os.path.join(asset, ANNOTATION_NAME)
        url = self._object_url(asset, fields="name")
        try:
            urllib.request.urlopen(self._request(url, method="HEAD"))
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            raise

    def assets_exist(self, assets: list[str]) -> list[tuple[str, bool]]:
        """Check if multiple assets exist concurrently."""
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [(a, pool.submit(self.asset_exists, a)) for a in assets]
            return [(a, f.result()) for a, f in futures]

    def download(self, blob_name: str, local_path: str) -> int:
        """Stream a single blob to a local file. Returns bytes downloaded."""
        url = self._DOWNLOAD.format(bucket=self.name, obj=urllib.parse.quote(blob_name, safe=""))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        size = 0
        with urllib.request.urlopen(self._request(url)) as resp, open(local_path, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                size += len(chunk)
        return size

    def upload(self, local_path: str, blob_name: str) -> int:
        """Stream a local file to a blob. Returns bytes uploaded."""
        size = os.path.getsize(local_path)
        with open(local_path, "rb") as f:
            req = self._request(
                self._upload_url(blob_name),
                data=f,
                method="POST",
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                },
            )
            urllib.request.urlopen(req)
        return size


def fetch_database(
    bucket: str,
    bucket_paths: list[str],
    destination: str,
    annotations: bool = True,
    model_formats: list[str] | None = None,
) -> None:
    """Download files from multiple paths in a Google Cloud Storage bucket.

    Args:
        bucket: The Google Cloud Storage bucket name to download from
        bucket_paths: List of paths inside the bucket to download
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

                if os.path.exists(local_file_path) and os.path.getsize(local_file_path) == blob_size:
                    logger.info(
                        "Skipping '%s' - already exists with correct size",
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
                        future.result()

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
