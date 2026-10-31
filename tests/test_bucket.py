import pytest

from arena_assets.impl.fetch import Bucket, outermost

API = "https://storage.googleapis.com/storage/v1/b/bkt/o"


@pytest.fixture(autouse=True)
def _default_endpoint(monkeypatch):
    monkeypatch.delenv("ARENA_ASSETS_ENDPOINT", raising=False)


def test_endpoint_override_reaches_every_url(monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_ENDPOINT", "http://127.0.0.1:9")
    bucket = Bucket("bkt")
    assert bucket._object_url("") == "http://127.0.0.1:9/storage/v1/b/bkt/o"
    assert bucket._upload_url("a").startswith("http://127.0.0.1:9/upload/")


def test_object_url_bare():
    assert Bucket("bkt")._object_url("") == API


def test_object_url_params():
    assert Bucket("bkt")._object_url("", prefix="a/") == f"{API}?prefix=a%2F"


def test_object_url_encodes_object():
    assert Bucket("bkt")._object_url("dir/file.yaml", fields="name") == f"{API}/dir%2Ffile.yaml?fields=name"


def test_upload_url_encodes_object():
    assert Bucket("bkt")._upload_url("dir/file.yaml") == "https://storage.googleapis.com/upload/storage/v1/b/bkt/o?uploadType=media&name=dir%2Ffile.yaml"


def test_request_carries_token():
    req = Bucket("bkt", token="tok")._request(API)
    assert req.get_header("Authorization") == "Bearer tok"


def test_request_anonymous_without_token():
    req = Bucket("bkt")._request(API)
    assert not req.has_header("Authorization")


def test_outermost_drops_link_bundles_of_articulated_objects():
    listed = ["Common/Object/Fridge", "Common/Object/Fridge/links/door", "Common/Object/Fridge/links/base", "Common/Object/Chair", "Common/Object/Fridge_B"]
    assert outermost(listed) == ["Common/Object/Fridge", "Common/Object/Chair", "Common/Object/Fridge_B"]
