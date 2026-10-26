import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from arena_assets.impl import DATABASE_NAME, Annotation
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.utils.Database import Database
from arena_assets.utils import logging as assets_logging
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.ModelConverter import BlenderUnavailable, command, environment
from http_host import Host


@pytest.fixture(autouse=True)
def _rebind_logging_to_stderr():
    yield
    assets_logging.initialize()


MATERIALS = [
    Annotation(
        name="OakWood",
        path="materials/oak",
        desc="light brown oak wood planks",
        tags=["wood"],
    ),
    Annotation(name="RedBrick", path="materials/brick", desc="red clay brick wall"),
    Annotation(name="SteelPlate", path="materials/steel", desc="brushed steel metal plate"),
]

OBJECTS = [
    ObjectAnnotation(
        name="OfficeChair",
        path="objects/chair",
        desc="ergonomic office chair",
        tags=["chair"],
        bounding_box=BoundingBox(((-0.3, 0.3), (-0.3, 0.3), (0.0, 0.9))),
    ),
    ObjectAnnotation(
        name="BarStool",
        path="objects/stool",
        desc="tall wooden bar stool",
        tags=["chair"],
        bounding_box=BoundingBox(((-0.2, 0.2), (-0.2, 0.2), (0.0, 1.1))),
    ),
]


@pytest.fixture(scope="session")
def database_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    db = Database(path / DATABASE_NAME)
    for annotation in MATERIALS:
        db.store("material", annotation)
    for annotation in OBJECTS:
        db.store("object", annotation)
    return path


@pytest.fixture(scope="session")
def blender() -> list[str]:
    try:
        return command()
    except BlenderUnavailable as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def blender_python(blender: list[str]) -> str:
    """Interpreter of the Blender worker, for scripts that build or inspect a scene: the shebang of a worker script, else the worker command itself."""
    program = Path(shutil.which(blender[0]) or blender[0])
    with open(program, "rb") as f:
        head = f.readline()
    interpreter = head[2:].decode().strip() if head.startswith(b"#!") else str(program)
    if subprocess.run([interpreter, "-c", "import bpy, arena_assets_blender"], capture_output=True, env=environment()).returncode:
        pytest.skip(f"{interpreter} from the Blender worker {blender} cannot import bpy and arena_assets_blender")
    return interpreter


CUBE = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 5 1 4 8
"""


@pytest.fixture()
def cube(tmp_path):
    path = tmp_path / "source" / "cube.obj"
    path.parent.mkdir()
    path.write_text(CUBE)
    return path


@pytest.fixture
def host():
    server = Host()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()
