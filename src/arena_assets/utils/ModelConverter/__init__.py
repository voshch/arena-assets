import enum


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


def sdf_model(stem: str) -> str:
    """Static SDF model whose visual and collision are both the sibling `{stem}.dae`."""
    return f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="{stem}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>{stem}.dae</uri>
          </mesh>
        </geometry>
      </visual>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>{stem}.dae</uri>
          </mesh>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>"""
