import dataclasses
import json
import os
import sys
import typing

import numpy

OPERATIONS = ("load", "rectify", "bounding_box", "lights", "remove_lights", "emissive_materials", "triangles", "limit_textures", "prepare_glow", "bind_uv_maps", "render_perspective", "render_topdown", "save", "split_links", "link_bounding_boxes", "save_link")


def convert(*args: str) -> None:
    from .converter import ModelConverter

    with ModelConverter() as converter:
        converter.load(args[0])
        converter.rectify()
        if len(args) >= 2:
            converter.save(args[1])


def encode(value: object) -> list | None:
    """JSON form of an operation's value: a list of records as dicts, a tuple of arrays as nested lists, a bounding box as its sides."""
    if value is None:
        return None
    if isinstance(value, list):
        return [dataclasses.asdict(item) for item in value]
    if isinstance(value, tuple) and all(isinstance(part, numpy.ndarray) for part in value):
        return [part.tolist() for part in value]
    return [list(side) for side in typing.cast("typing.Iterable[typing.Iterable[float]]", value)]


class Scene:
    """Converter scene of the current request, kept open for the next request when asked."""

    def __init__(self) -> None:
        self._converter: typing.Any = None
        self._reported = (0, 0)

    def run(self, steps: list[list], keep: bool = False) -> dict:
        """Run [operation, *args, kwargs?] steps in order, in a fresh scene unless the last request kept its own."""
        from .converter import ModelConverter

        if self._converter is None:
            self._converter = ModelConverter().__enter__()
            self._reported = (0, 0)
        converter = self._converter
        values: list = []
        error = None
        try:
            for operation, *args in steps:
                if operation not in OPERATIONS:
                    raise ValueError(f"unknown operation {operation!r}, known: {list(OPERATIONS)}")
                kwargs = args.pop() if args and isinstance(args[-1], dict) else {}
                values.append(encode(getattr(converter, operation)(*args, **kwargs)))
        except Exception as e:
            error = repr(e)
        if error is not None or not keep:
            converter.__exit__(None, None, None)
            self._converter = None
        stdout, stderr = converter.stdout, converter.stderr
        reply = {"ok": error is None, "values": values, "stdout": stdout[self._reported[0] :], "stderr": stderr[self._reported[1] :]}
        self._reported = (len(stdout), len(stderr))
        if error is not None:
            reply["error"] = error
        return reply


def serve(requests: typing.TextIO, replies: typing.TextIO) -> None:
    """Answer one JSON reply line per JSON request line ({"steps": [...], "keep": false}) until the requests end."""
    scene = Scene()
    for line in requests:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            reply = scene.run(request["steps"], keep=request.get("keep", False))
        except Exception as e:
            reply = {"ok": False, "values": [], "stdout": "", "stderr": "", "error": repr(e)}
        replies.write(json.dumps(reply) + "\n")
        replies.flush()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        name = os.path.basename(sys.argv[0])
        print(f"Usage: {name} <input_file> [<output_file>]\n       {name} --exts\n       {name} --serve")
        return
    if args == ["--exts"]:
        from .converter import ModelConverter

        print(json.dumps(list(ModelConverter.exts())))
        return
    if args == ["--serve"]:
        replies = os.fdopen(os.dup(sys.stdout.fileno()), "w")
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        serve(sys.stdin, replies)
        return
    convert(*args)


if __name__ == "__main__":
    main()
