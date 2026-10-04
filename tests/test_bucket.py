from arena_assets.impl.fetch import Bucket

API = "https://storage.googleapis.com/storage/v1/b/bkt/o"


def test_object_url_bare():
    assert Bucket("bkt")._object_url("") == API


def test_object_url_params():
    assert Bucket("bkt")._object_url("", prefix="a/") == f"{API}?prefix=a%2F"


def test_object_url_encodes_object():
    assert (
        Bucket("bkt")._object_url("dir/file.yaml", fields="name")
        == f"{API}/dir%2Ffile.yaml?fields=name"
    )


def test_upload_url_encodes_object():
    assert (
        Bucket("bkt")._upload_url("dir/file.yaml")
        == "https://storage.googleapis.com/upload/storage/v1/b/bkt/o?uploadType=media&name=dir%2Ffile.yaml"
    )


def test_request_carries_token():
    req = Bucket("bkt", token="tok")._request(API)
    assert req.get_header("Authorization") == "Bearer tok"


def test_request_anonymous_without_token():
    req = Bucket("bkt")._request(API)
    assert not req.has_header("Authorization")
