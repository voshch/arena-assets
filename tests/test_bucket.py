import hashlib
import json
import os
import sqlite3
import time
import urllib.parse

import pytest

from arena_assets.impl.fetch import DEFAULT_URL, MAX_UPLOAD, Bucket, StoreError, cache_root, fetch_database, outermost
from arena_assets.utils.logging import initialize


@pytest.fixture
def private(store, monkeypatch):
    store.private.add("arena-internal")
    store.tokens["good"] = "alice"
    for name in ["w1/world.yaml", "w2/0/world.yaml", "w2/1/world.yaml", "world.yaml", "Common/Object/a/annotation.yaml", "recordings/take.mp4"]:
        store.put("arena-internal", name, name.encode())
    return store


def test_default_url_without_override(monkeypatch):
    monkeypatch.delenv("ARENA_ASSETS_URL", raising=False)
    assert Bucket("public/models")._object_url("a") == f"{DEFAULT_URL}/public/models/a"


def test_url_override_reaches_objects_and_index(store):
    store.put("public", "models/a/x.obj", b"1")
    bucket = Bucket("public/models")
    assert bucket._object_url("a/x.obj") == f"{store.base}/public/models/a/x.obj"
    bucket.listdir("")
    assert store.seen("GET", "/public/index.json")


def test_object_url_encodes_segments_and_keeps_slashes(store):
    assert Bucket("arena-internal")._object_url("dir/a b#.yaml") == f"{store.base}/arena-internal/dir/a%20b%23.yaml"


def test_cache_root_falls_back_to_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("ARENA_ASSETS_DIR", raising=False)
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    assert cache_root() == tmp_path / "assets"


def test_no_cache_directory_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ARENA_ASSETS_DIR", raising=False)
    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    with pytest.raises(StoreError, match="set ARENA_DATA_DIR or ARENA_ASSETS_DIR"):
        cache_root()


def test_request_carries_token():
    assert Bucket._request("http://x", "tok").get_header("Authorization") == "Bearer tok"


def test_request_anonymous_without_token():
    assert not Bucket._request("http://x").has_header("Authorization")


def test_outermost_drops_link_bundles_of_articulated_objects():
    listed = ["Common/Object/Fridge", "Common/Object/Fridge/links/door", "Common/Object/Fridge/links/base", "Common/Object/Chair", "Common/Object/Fridge_B"]
    assert outermost(listed) == ["Common/Object/Fridge", "Common/Object/Chair", "Common/Object/Fridge_B"]


def test_public_folder_is_stripped_from_names(store):
    store.put("public", "models/Common/Object/a/annotation.yaml", b"a")
    store.put("public", "worlds/w1/world.yaml", b"w")
    models = Bucket("public/models")
    assert [entry["name"] for entry in models.listdir("")] == ["Common/Object/a/annotation.yaml"]
    assert models.list_assets() == ["Common/Object/a"]
    assert Bucket("public/worlds").listchildren("") == ["w1"]


def test_index_is_fetched_once_per_ttl_across_processes(store):
    store.put("public", "models/a/x.obj", b"1")
    Bucket("public/models").listdir("a")
    Bucket("public/models").object_exists("a/x.obj")
    assert len(store.seen("GET", "/public/index.json")) == 1


def test_index_is_requested_gzip_compressed(store):
    store.put("public", "models/a/x.obj", b"1")
    assert Bucket("public/models").object_exists("a/x.obj")
    assert store.seen("GET", "/public/index.json")[0]["Accept-Encoding"] == "gzip"


def test_expired_index_revalidates_with_its_etag(store, monkeypatch):
    store.put("public", "models/a/x.obj", b"1")
    Bucket("public/models").listdir("a")
    monkeypatch.setenv("ARENA_MODELS_TTL", "0")
    assert Bucket("public/models").object_exists("a/x.obj")
    second = store.seen("GET", "/public/index.json")[1]
    assert second["If-None-Match"] == f'"public-{store.generation}"'


def test_a_miss_revalidates_a_cached_index_once(store):
    store.put("public", "models/a/x.obj", b"1")
    Bucket("public/models").listdir("a")
    store.put("public", "models/b/y.obj", b"2")
    bucket = Bucket("public/models")
    assert bucket.object_exists("b/y.obj")
    assert not bucket.object_exists("c/z.obj")
    assert len(store.seen("GET", "/public/index.json")) == 2


def test_unknown_bucket_names_the_store(store):
    with pytest.raises(StoreError, match="no bucket 'arena-worlds-prod-public' on .*, known buckets are public, arena-internal"):
        Bucket("arena-worlds-prod-public").listdir("")


def test_refused_read_retries_with_the_environment_token(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    assert len(Bucket("arena-internal").listdir("")) == 6
    assert [headers.get("Authorization") for headers in private.seen("GET", "/arena-internal/index.json")] == [None, "Bearer good"]


def test_token_sticks_after_the_first_refusal(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    bucket = Bucket("arena-internal")
    bucket.listdir("")
    bucket.download("world.yaml", str(cache_root() / "out" / "world.yaml"))
    assert private.seen("GET", "/arena-internal/world.yaml")[0]["Authorization"] == "Bearer good"


def test_refused_read_without_a_token_names_the_way_to_get_one(private):
    with pytest.raises(StoreError, match="GET arena-internal/index.json: 401 bearer token required, set ARENA_ASSETS_TOKEN") as refused:
        Bucket("arena-internal").listdir("")
    assert refused.value.status == 401


def _gcloud_login(config, active: str, credentials: dict[str, dict]) -> None:
    (config / "configurations").mkdir(parents=True)
    (config / "active_config").write_text("work")
    (config / "configurations" / "config_work").write_text(f"[core]\naccount = {active}\n")
    with sqlite3.connect(config / "credentials.db") as db:
        db.execute("CREATE TABLE credentials (account_id TEXT PRIMARY KEY, value BLOB)")
        db.executemany("INSERT INTO credentials VALUES (?, ?)", [(account, json.dumps(value)) for account, value in credentials.items()])
    db.close()


def test_refused_read_refreshes_a_token_from_the_stored_gcloud_login(private, host, monkeypatch, tmp_path):
    host.pages["/token"] = json.dumps({"access_token": "good", "expires_in": 3599}).encode()
    login = {"type": "authorized_user", "client_id": "cli", "client_secret": "s", "refresh_token": "r-me", "token_uri": f"{host.base}/token"}
    _gcloud_login(tmp_path / "gcloud", "me@example.org", {"me@example.org": login, "other@example.org": {**login, "refresh_token": "r-other"}})
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path / "gcloud"))
    Bucket("arena-internal").listdir("")
    assert private.seen("GET", "/arena-internal/index.json")[-1]["Authorization"] == "Bearer good"
    assert urllib.parse.parse_qs(host.posted["/token"].decode()) == {"grant_type": ["refresh_token"], "client_id": ["cli"], "client_secret": ["s"], "refresh_token": ["r-me"]}


def test_stored_gcloud_login_without_the_active_account_names_the_way_to_get_one(private, monkeypatch, tmp_path):
    _gcloud_login(tmp_path / "gcloud", "me@example.org", {"other@example.org": {"client_id": "cli", "client_secret": "s", "refresh_token": "r"}})
    monkeypatch.setenv("CLOUDSDK_CONFIG", str(tmp_path / "gcloud"))
    with pytest.raises(StoreError, match="401 bearer token required, set ARENA_ASSETS_TOKEN or log in with 'gcloud auth login'"):
        Bucket("arena-internal").listdir("")


def test_invalid_token_raises_after_one_retry(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "stale")
    with pytest.raises(StoreError, match="not accepted") as refused:
        Bucket("arena-internal").listdir("")
    assert refused.value.status == 401
    assert [headers.get("Authorization") for headers in private.seen("GET", "/arena-internal/index.json")] == [None, "Bearer stale"]


def test_explicit_token_is_not_replaced_on_refusal(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    with pytest.raises(StoreError):
        Bucket("arena-internal", token="stale").listdir("")
    assert [headers.get("Authorization") for headers in private.seen("GET", "/arena-internal/index.json")] == ["Bearer stale"]


def test_missing_object_stays_false_on_a_private_bucket(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    assert Bucket("arena-internal").object_exists("absent.yaml") is False


def test_listchildren_containing_keeps_only_children_with_that_file(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    assert Bucket("arena-internal").listchildren("", containing="world.yaml") == ["w1", "w2"]


def test_asset_exists_without_sentinel_takes_a_prefix_or_an_object(private, monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    bucket = Bucket("arena-internal")
    assert bucket.assets_exist(["w2", "world.yaml", "w3"], sentinel=None) == [("w2", True), ("world.yaml", True), ("w3", False)]
    assert bucket.asset_exists("Common/Object/a")


def test_download_resumes_a_partial_file_with_a_range(store, tmp_path):
    store.put("public", "models/a/x.bin", b"0123456789")
    target = tmp_path / "a" / "x.bin"
    target.parent.mkdir()
    (tmp_path / "a" / "x.bin.part").write_bytes(b"0123")
    assert Bucket("public/models").download("a/x.bin", str(target)) == 6
    assert target.read_bytes() == b"0123456789"
    assert store.seen("GET", "/public/models/a/x.bin")[0]["Range"] == "bytes=4-"


def test_download_of_a_complete_partial_file_needs_no_body(store, tmp_path):
    store.put("public", "models/x.bin", b"0123")
    (tmp_path / "x.bin.part").write_bytes(b"0123")
    assert Bucket("public/models").download("x.bin", str(tmp_path / "x.bin")) == 0
    assert (tmp_path / "x.bin").read_bytes() == b"0123"


def test_corrupt_partial_file_is_downloaded_again(store, tmp_path):
    store.put("public", "models/x.bin", b"0123456789")
    (tmp_path / "x.bin.part").write_bytes(b"XXXX")
    Bucket("public/models").download("x.bin", str(tmp_path / "x.bin"))
    assert (tmp_path / "x.bin").read_bytes() == b"0123456789"
    assert [headers.get("Range") for headers in store.seen("GET", "/public/models/x.bin")] == ["bytes=4-", None]


def test_download_not_matching_the_index_raises_and_keeps_nothing(store, tmp_path):
    store.put("public", "models/x.bin", b"old")
    bucket = Bucket("public/models")
    bucket.objects()
    store.buckets["public"]["models/x.bin"] = b"new"
    with pytest.raises(StoreError, match="does not match"):
        bucket.download("x.bin", str(tmp_path / "x.bin"))
    assert not list(tmp_path.glob("x.bin*"))


def test_download_after_a_replacement_refreshes_the_index(store, tmp_path):
    store.put("public", "models/x.bin", b"old")
    Bucket("public/models").objects()
    store.put("public", "models/x.bin", b"new")
    Bucket("public/models").download("x.bin", str(tmp_path / "x.bin"))
    assert (tmp_path / "x.bin").read_bytes() == b"new"


def test_upload_sends_the_token_and_the_content_hash(store, tmp_path):
    store.tokens["tok"] = "alice"
    source = tmp_path / "x.bin"
    source.write_bytes(b"payload")
    assert Bucket("public/models", token="tok").upload(str(source), "a/x.bin") == 7
    assert store.buckets["public"]["models/a/x.bin"] == b"payload"
    headers = store.seen("PUT", "/public/models/a/x.bin")[0]
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X-Content-SHA256"] == hashlib.sha256(b"payload").hexdigest()
    assert "If-None-Match" not in headers


def test_upload_without_overwrite_is_refused_on_an_existing_object(store, tmp_path):
    store.tokens["tok"] = "alice"
    store.put("public", "models/x.bin", b"theirs")
    source = tmp_path / "x.bin"
    source.write_bytes(b"mine")
    with pytest.raises(StoreError, match="PUT public/models/x.bin: 412 exists, it exists with other content and --no-overwrite keeps it"):
        Bucket("public/models", token="tok").upload(str(source), "x.bin", overwrite=False)
    assert store.seen("PUT", "/public/models/x.bin")[0]["If-None-Match"] == "*"
    assert store.buckets["public"]["models/x.bin"] == b"theirs"


def test_refused_upload_names_the_owner(store, tmp_path):
    store.tokens["tok"] = "alice"
    store.owners[("public", "models/x.bin")] = "bob"
    source = tmp_path / "x.bin"
    source.write_bytes(b"mine")
    with pytest.raises(StoreError, match="403 public/models/x.bin belongs to bob, .*/_api/whoami shows"):
        Bucket("public/models", token="tok").upload(str(source), "x.bin")


def test_upload_resolves_the_environment_token(store, tmp_path, monkeypatch):
    store.tokens["env"] = "alice"
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "env")
    source = tmp_path / "x.bin"
    source.write_bytes(b"1")
    Bucket("arena-internal").upload(str(source), "x.bin")
    assert store.seen("PUT", "/arena-internal/x.bin")[0]["Authorization"] == "Bearer env"


def test_upload_over_the_size_limit_never_reaches_the_store(store, tmp_path):
    source = tmp_path / "huge.bin"
    with open(source, "wb") as f:
        f.truncate(MAX_UPLOAD + 1)
    with pytest.raises(StoreError, match="at most"):
        Bucket("public/models", token="tok").upload(str(source), "huge.bin", sha256="0" * 64)
    assert not store.seen("PUT", "/public/models/huge.bin")


def test_a_write_makes_the_next_process_revalidate_the_index(store, tmp_path):
    store.tokens["tok"] = "alice"
    Bucket("public/models").listdir("")
    source = tmp_path / "x.bin"
    source.write_bytes(b"1")
    Bucket("public/models", token="tok").upload(str(source), "x.bin")
    assert os.path.getmtime(cache_root() / ".index" / "public.json") < time.time() - 86400
    assert Bucket("public/models").object_exists("x.bin")


def test_delete_removes_an_own_object(store):
    store.tokens["tok"] = "alice"
    store.put("public", "models/x.bin", b"1")
    store.owners[("public", "models/x.bin")] = "alice"
    Bucket("public/models", token="tok").delete("x.bin")
    assert "models/x.bin" not in store.buckets["public"]
    assert not Bucket("public/models").object_exists("x.bin")


def test_delete_of_another_users_object_is_refused(store):
    store.tokens["tok"] = "alice"
    store.put("public", "models/x.bin", b"1")
    store.owners[("public", "models/x.bin")] = "bob"
    with pytest.raises(StoreError, match="403.*bob"):
        Bucket("public/models", token="tok").delete("x.bin")


def test_unreachable_store_says_so_in_one_line(monkeypatch):
    monkeypatch.setenv("ARENA_ASSETS_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARENA_ASSETS_DIR", "/nonexistent-cache")
    with pytest.raises(StoreError, match="cannot reach the asset store at http://127.0.0.1:9"):
        Bucket("public/models").listdir("")


def test_object_gone_since_the_index_names_it(store, tmp_path):
    store.put("public", "models/x.bin", b"1")
    bucket = Bucket("public/models")
    bucket.objects()
    del store.buckets["public"]["models/x.bin"]
    with pytest.raises(StoreError, match="GET public/models/x.bin: 404 no such object") as missing:
        bucket.download("x.bin", str(tmp_path / "x.bin"))
    assert missing.value.status == 404


def test_browse_shows_folders_with_counts_and_files_at_one_level(store):
    store.put("public", "models/Common/Object/a/annotation.yaml", b"12")
    store.put("public", "models/Common/Object/a/a.usdz", b"3456")
    store.put("public", "models/Common/Object/b/annotation.yaml", b"7")
    store.put("public", "models/Common/readme.md", b"89")
    folders, files = Bucket("public/models").browse("Common")
    assert folders == {"Object": (3, 7)}
    assert [entry["name"] for entry in files] == ["readme.md"]
    assert Bucket("public/models").browse("Common/Object/") == ({"a": (2, 6), "b": (1, 1)}, [])


def test_browse_of_a_bucket_root_spans_its_folders(store):
    store.put("public", "models/a.txt", b"1")
    store.put("public", "worlds/w/world.yaml", b"22")
    assert Bucket("public").browse() == ({"models": (1, 1), "worlds": (1, 2)}, [])


def test_under_selects_a_folder_and_never_a_whole_source(store):
    for name in ["w/world.yaml", "w/0/map.png", "w2/world.yaml", "w"]:
        store.put("arena-internal", name, b"x")
    bucket = Bucket("arena-internal", token="tok")
    assert bucket.under("w/") == ["w", "w/0/map.png", "w/world.yaml"]
    with pytest.raises(StoreError, match="whole source"):
        bucket.under("/")


def test_parallel_downloads_from_a_private_bucket_share_one_token(private, monkeypatch, tmp_path):
    initialize(silent=True)
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "good")
    for n in range(40):
        private.put("arena-internal", f"many/{n}.bin", str(n).encode())
    Bucket("arena-internal").objects()
    fetch_database("arena-internal", ["many"], str(tmp_path))
    assert sorted(p.name for p in (tmp_path / "many").iterdir()) == sorted(f"{n}.bin" for n in range(40))


def test_delete_of_an_object_already_gone_counts_as_removed(store):
    store.tokens["tok"] = "alice"
    store.buckets["arena-internal"] = {}
    Bucket("arena-internal", token="tok").delete("gone.bin")
    assert store.seen("DELETE", "/arena-internal/gone.bin")
