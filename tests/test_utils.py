from arena_assets.impl import convert_list_str
from arena_assets.utils.logging import format_file_size


def test_format_file_size():
    assert format_file_size(0) == "0B"
    assert format_file_size(512) == "512B"
    assert format_file_size(1024) == "1.0kB"
    assert format_file_size(1536) == "1.5kB"
    assert format_file_size(1024**2) == "1.0MB"
    assert format_file_size(5.5 * 1024**3) == "5.5GB"


def test_convert_list_str():
    assert convert_list_str(None) == []
    assert convert_list_str(["a", "b"]) == ["a", "b"]
    assert convert_list_str("a") == ["a"]
