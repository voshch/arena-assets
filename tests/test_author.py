import pytest

from arena_assets.impl import author
from arena_assets.impl.author import author_database
from arena_assets.utils.logging import initialize


@pytest.fixture
def bucket(monkeypatch):
    initialize(silent=True)

    class FakeBucket:
        blobs: dict[str, bytes] = {}
        uploads: list[tuple[str, str]] = []

        def __init__(self, name, token=None):
            self.name = name
            self.token = token

        def listdir(self, prefix):
            prefix = prefix.strip("/")
            if prefix:
                prefix += "/"
            return [
                {"name": name, "size": str(len(data))}
                for name, data in self.blobs.items()
                if name.startswith(prefix)
            ]

        def upload(self, local_path, blob_name):
            with open(local_path, "rb") as f:
                data = f.read()
            self.uploads.append((local_path, blob_name))
            self.blobs[blob_name] = data
            return len(data)

    monkeypatch.setattr(author, "Bucket", FakeBucket)
    monkeypatch.setattr(author, "resolve_token", lambda: "token")
    return FakeBucket


def make_database(tmp_path, files):
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_files_land_under_destination(bucket, tmp_path):
    make_database(
        tmp_path,
        {
            "a/annotation.yaml": b"a: 1",
            "a/model.obj": b"obj-data",
        },
    )
    author_database("bkt", str(tmp_path), "assets")
    assert bucket.blobs == {
        "assets/a/annotation.yaml": b"a: 1",
        "assets/a/model.obj": b"obj-data",
    }


def test_no_destination_uploads_to_root(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"1"})
    author_database("bkt", str(tmp_path))
    assert [name for _, name in bucket.uploads] == ["a/model.obj"]


def test_skip_existing_with_same_size(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"12345"})
    bucket.blobs = {"a/model.obj": b"54321"}
    author_database("bkt", str(tmp_path))
    assert bucket.uploads == []
    assert bucket.blobs["a/model.obj"] == b"54321"


def test_reupload_on_size_mismatch(bucket, tmp_path):
    make_database(tmp_path, {"a/model.obj": b"123456"})
    bucket.blobs = {"a/model.obj": b"12"}
    author_database("bkt", str(tmp_path))
    assert bucket.blobs["a/model.obj"] == b"123456"


def test_empty_source_uploads_nothing(bucket, tmp_path):
    author_database("bkt", str(tmp_path))
    assert bucket.uploads == []


def test_upload_error_propagates(bucket, tmp_path):
    make_database(tmp_path, {"a/x.obj": b"1", "a/y.obj": b"22"})

    def boom(self, local_path, blob_name):
        raise OSError(f"boom {blob_name}")

    bucket.upload = boom
    with pytest.raises(OSError):
        author_database("bkt", str(tmp_path))


def test_explicit_token_skips_resolution(bucket, tmp_path, monkeypatch):
    def fail():
        raise AssertionError("resolve_token should not be called")

    monkeypatch.setattr(author, "resolve_token", fail)
    make_database(tmp_path, {"a/model.obj": b"1"})
    author_database("bkt", str(tmp_path), token="explicit")
