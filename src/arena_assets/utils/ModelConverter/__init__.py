import enum
from collections.abc import Sequence


class ModelFormat(enum.StrEnum):
    OBJ = "obj"
    FBX = "fbx"
    USDZ = "usdz"
    USDA = "usda"
    USDC = "usdc"
    USD = "usd"
    DAE = "dae"
    GLTF = "gltf"
    GLB = "glb"
    SDF = "sdf"


def _mesh_element(tag: str, name: str, uri: str, material: str = "") -> str:
    return f"""      <{tag} name="{name}">
        <geometry>
          <mesh>
            <uri>{uri}</uri>
          </mesh>
        </geometry>{material}
      </{tag}>"""


def sdf_model(stem: str, glow: Sequence[tuple[str, str, str]] = (), *, plain: bool = True) -> str:
    """Static SDF model whose visual and collision are the sibling `{stem}.dae`, plus a visual and a collision per glow mesh (name, uri, material)."""
    parts = [_mesh_element("visual", "visual", f"{stem}.dae"), _mesh_element("collision", "collision", f"{stem}.dae")] if plain else []
    for index, (name, uri, material) in enumerate(glow):
        parts += [_mesh_element("visual", name, uri, material), _mesh_element("collision", f"collision_glow_{index}", uri)]
    link = "\n".join(parts)
    return f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{stem}">
    <static>true</static>
    <link name="link">
{link}
    </link>
  </model>
</sdf>"""
