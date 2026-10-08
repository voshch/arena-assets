import os

import pytest

from arena_assets.impl import fetch
from arena_assets.impl.fetch import fetch_database
from arena_assets.utils.logging import initialize


@pytest.fixture
def bucket(monkeypatch):
    initialize(silent=True)

    class FakeBucket:
        blobs: dict[str, bytes] = {}
        downloads: list[tuple[str, str]] = []

        def __init__(self, name):
            self.name = name

        def listdirs(self, prefixes):
            result = {}
            for path in prefixes:
                prefix = path.strip("/")
                if prefix:
                    prefix += "/"
                result[path] = [{"name": name, "size": str(len(data))} for name, data in self.blobs.items() if name.startswith(prefix)]
            return result

        def download(self, blob_name, local_path):
            self.downloads.append((blob_name, local_path))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(self.blobs[blob_name])
            return len(self.blobs[blob_name])

    monkeypatch.setattr(fetch, "Bucket", FakeBucket)
    return FakeBucket


def test_files_land_under_destination(bucket, tmp_path):
    bucket.blobs = {
        "assets/a/annotation.yaml": b"a: 1",
        "assets/a/model.obj": b"obj-data",
    }
    fetch_database("bkt", ["assets/a"], str(tmp_path))
    assert (tmp_path / "assets/a/annotation.yaml").read_bytes() == b"a: 1"
    assert (tmp_path / "assets/a/model.obj").read_bytes() == b"obj-data"


def test_absolute_blob_name_stays_contained(bucket, tmp_path):
    bucket.blobs = {"/etc/passwd": b"pwned"}
    fetch_database("bkt", [""], str(tmp_path))
    assert bucket.downloads == [("/etc/passwd", str(tmp_path / "etc/passwd"))]


def test_format_filter(bucket, tmp_path):
    bucket.blobs = {
        "a/model.obj": b"1",
        "a/model.fbx": b"2",
    }
    fetch_database("bkt", ["a"], str(tmp_path), model_formats=["obj"])
    assert [name for name, _ in bucket.downloads] == ["a/model.obj"]


def test_no_annotations(bucket, tmp_path):
    bucket.blobs = {
        "a/annotation.yaml": b"a: 1",
        "a/model.obj": b"1",
    }
    fetch_database("bkt", ["a"], str(tmp_path), annotations=False)
    assert [name for name, _ in bucket.downloads] == ["a/model.obj"]


def test_skip_existing_with_same_size(bucket, tmp_path):
    bucket.blobs = {"a/model.obj": b"12345"}
    target = tmp_path / "a/model.obj"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"54321")
    fetch_database("bkt", ["a"], str(tmp_path))
    assert bucket.downloads == []
    assert target.read_bytes() == b"54321"


def test_redownload_on_size_mismatch(bucket, tmp_path):
    bucket.blobs = {"a/model.obj": b"123456"}
    target = tmp_path / "a/model.obj"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"12")
    fetch_database("bkt", ["a"], str(tmp_path))
    assert target.read_bytes() == b"123456"


def test_dedup_overlapping_paths(bucket, tmp_path):
    bucket.blobs = {"a/b/model.obj": b"1"}
    fetch_database("bkt", ["a", "a/b"], str(tmp_path))
    assert len(bucket.downloads) == 1


def test_download_error_propagates(bucket, tmp_path):
    bucket.blobs = {"a/x.obj": b"1", "a/y.obj": b"22"}

    def boom(self, blob_name, local_path):
        raise OSError(f"boom {blob_name}")

    bucket.download = boom
    with pytest.raises(OSError):
        fetch_database("bkt", ["a"], str(tmp_path))
