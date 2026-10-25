from pathlib import Path

import numpy as np

URDF = """<robot name="cabinet">
  <link name="base"/>
  <link name="link_1"><visual><origin xyz="0 0 0"/><geometry><mesh filename="textured_objs/body.obj"/></geometry></visual></link>
  <link name="link_0">
    <visual><origin xyz="-0.5 0 0"/><geometry><mesh filename="textured_objs/door.obj"/></geometry></visual>
    <visual><origin xyz="-0.5 0 0"/><geometry><mesh filename="textured_objs/body.obj"/></geometry></visual>
  </link>
  <joint name="joint_1" type="fixed"><origin rpy="1.570796326794897 0 -1.570796326794897" xyz="0 0 0"/><child link="link_1"/><parent link="base"/></joint>
  <joint name="joint_0" type="revolute"><origin xyz="0.5 0 0"/><axis xyz="0 1 0"/><child link="link_0"/><parent link="link_1"/><limit lower="0" upper="1.57"/></joint>
</robot>
"""
OBJ = """mtllib {name}.mtl
v 0 0 0
v 0.2 0 0
v 0.2 0.4 0
v 0 0.4 -0.1
vt 0 0
vt 1 0
vt 1 1
vn 0 0 1
usemtl wood
f 1/1/1 2/2/1 3/3/1
f -4/-3/-1 -2/-1/-1 -1/-1/-1
"""
UP = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def urdf_item(folder: Path) -> Path:
    """A PartNet-Mobility style URDF: a Y-up mesh frame stood up by the root joint, a door hinged off the body, two visuals sharing a material name."""
    (folder / "textured_objs").mkdir(parents=True)
    (folder / "images").mkdir()
    (folder / "images" / "wood.png").write_bytes(b"png")
    for name in ("body", "door"):
        (folder / "textured_objs" / f"{name}.obj").write_text(OBJ.format(name=name))
        (folder / "textured_objs" / f"{name}.mtl").write_text("newmtl wood\nillum 3\nKd 0.5 0.4 0.3\nmap_Kd ../images/wood.png\n")
    (folder / "mobility.urdf").write_text(URDF)
    return folder / "mobility.urdf"
