import io
import json
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

SCENE = """import json, sys, bpy
from arena_assets_blender.converter import ModelConverter

big, deep, small, packed, out, usdz = sys.argv[1:]
with ModelConverter() as converter:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    material = bpy.data.materials.new("Skin")
    material.use_nodes = True
    surface = material.node_tree.nodes["Principled BSDF"]
    for path, socket in ((big, "Base Color"), (deep, None), (small, None), (packed, "Roughness")):
        node = material.node_tree.nodes.new("ShaderNodeTexImage")
        node.image = bpy.data.images.load(path)
        if socket is not None:
            material.node_tree.links.new(node.outputs["Color"], surface.inputs[socket])
    bpy.data.images[packed.rsplit("/", 1)[-1]].pack()
    bpy.context.scene.objects[0].data.materials.append(material)
    before = {image.name: (tuple(image.size), image.is_float) for image in bpy.data.images}
    changed = converter.limit_textures(2048, out)
    after = {image.name: (tuple(image.size), image.is_float, image.packed_file is not None) for image in bpy.data.images}
    converter.save(usdz)
print(json.dumps({"before": before, "after": after, "changed": sorted(limit.name for limit in changed)}))
"""


def _textures(tmp_path):
    paths = {name: tmp_path / f"{name}.png" for name in ("big", "deep", "small", "packed")}
    Image.new("RGB", (4096, 8), (200, 10, 10)).save(paths["big"])
    Image.fromarray(np.full((4, 4), 40000, dtype=np.uint16)).save(paths["deep"])
    Image.new("RGB", (64, 64), (10, 200, 10)).save(paths["small"])
    Image.new("RGBA", (8, 4096), (10, 10, 200, 255)).save(paths["packed"])
    return paths


def test_limit_textures_fits_file_and_packed_images_and_stores_16_bits_as_8(tmp_path):
    paths = _textures(tmp_path)
    proc = subprocess.run([sys.executable, "-c", SCENE, *(str(paths[name]) for name in ("big", "deep", "small", "packed")), str(tmp_path / "limited"), str(tmp_path / "skin.usdz")], capture_output=True, text=True, timeout=300, check=True)
    result = json.loads(proc.stdout.splitlines()[-1])
    assert result["before"]["big.png"] == [[4096, 8], False]
    assert result["before"]["deep.png"][1] is True
    assert result["changed"] == ["big.png", "deep.png", "packed.png"]
    assert result["after"]["big.png"] == [[2048, 4], False, False]
    assert result["after"]["deep.png"] == [[4, 4], False, False]
    assert result["after"]["small.png"] == [[64, 64], False, False]
    assert result["after"]["packed.png"] == [[4, 2048], False, True]


def test_limited_textures_are_what_the_exported_model_carries(tmp_path):
    paths = _textures(tmp_path)
    subprocess.run([sys.executable, "-c", SCENE, *(str(paths[name]) for name in ("big", "deep", "small", "packed")), str(tmp_path / "limited"), str(tmp_path / "skin.usdz")], capture_output=True, text=True, timeout=300, check=True)
    with zipfile.ZipFile(tmp_path / "skin.usdz") as usdz:
        sizes = sorted(Image.open(io.BytesIO(usdz.read(name))).size for name in usdz.namelist() if name.endswith((".png", ".jpg")))
    assert sizes == [(4, 2048), (2048, 4)]


PATHLESS = """import json, sys, bpy
from arena_assets_blender.converter import ModelConverter

texture, out = sys.argv[1:]
with ModelConverter() as converter:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    material = bpy.data.materials.new("Skin")
    material.use_nodes = True
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(texture)
    node.image.pack()
    node.image.filepath = ""
    material.node_tree.links.new(node.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    bpy.context.scene.objects[0].data.materials.append(material)
    converter.save(out)
    result = {"filepath": node.image.filepath, "packed": node.image.packed_file is not None}
print(json.dumps(result))
"""


def test_an_obj_names_the_packed_images_a_gltf_brings_without_a_file(tmp_path):
    Image.new("RGB", (8, 8), (200, 10, 10)).save(tmp_path / "red.png")
    out = tmp_path / "skin.obj"
    proc = subprocess.run([sys.executable, "-c", PATHLESS, str(tmp_path / "red.png"), str(out)], capture_output=True, text=True, timeout=300, check=True)
    assert json.loads(proc.stdout.splitlines()[-1]) == {"filepath": "", "packed": True}
    maps = [line.split()[-1] for line in (out / "skin.mtl").read_text().splitlines() if line.startswith("map_Kd")]
    assert maps
    assert all(Image.open(out / name).getpixel((0, 0)) == (200, 10, 10) for name in maps)


@pytest.mark.parametrize(
    ("mode", "size", "max_px", "expected", "image_format"),
    [("RGB", (4096, 100), 2048, (2048, 50), "JPEG"), ("RGBA", (300, 300), 2048, (300, 300), "PNG"), ("I;16", (20, 10), 8, (8, 4), "JPEG")],
)
def test_limited_image_fits_the_longer_side_with_8_bits_per_channel(mode, size, max_px, expected, image_format):
    from arena_assets_blender.textures import limited

    image = Image.fromarray(np.full(size[::-1], 50000, dtype=np.uint16)) if mode == "I;16" else Image.new(mode, size)
    data = io.BytesIO()
    image.save(data, format="PNG")
    encoded, encoded_format, before, after = limited(data.getvalue(), max_px)
    out = Image.open(io.BytesIO(encoded))
    assert (before, after, out.size, encoded_format, out.format) == (size, expected, expected, image_format, image_format)
    assert out.mode in ("RGB", "RGBA", "L")
    if mode == "I;16":
        assert abs(out.getpixel((0, 0)) - (50000 >> 8)) <= 1


def test_an_opaque_rgba_image_drops_its_alpha_and_becomes_jpeg():
    from arena_assets_blender.textures import limited

    data = io.BytesIO()
    Image.new("RGBA", (4096, 4), (10, 20, 30, 255)).save(data, format="PNG")
    encoded, image_format, _, after = limited(data.getvalue(), 2048)
    assert (image_format, after, Image.open(io.BytesIO(encoded)).mode) == ("JPEG", (2048, 2), "RGB")


def test_an_image_past_the_decompression_bomb_limit_is_fit_too():
    from arena_assets_blender.textures import limited

    data = io.BytesIO()
    Image.new("L", (13400, 13400)).save(data, format="PNG")
    _, image_format, before, after = limited(data.getvalue(), 2048)
    assert (image_format, before, after) == ("JPEG", (13400, 13400), (2048, 2048))
