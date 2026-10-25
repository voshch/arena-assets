# arena-assets

Build, query, and manage 3D model databases for [Arena](https://github.com/Arena-Rosnav).

`arena-assets` converts datasets of assets (objects, materials, humans, and sounds) into a searchable database: models are converted with Blender, optionally baked with Isaac Sim, and indexed in [ChromaDB](https://www.trychroma.com/) so they can be queried by natural-language description.

## Installation

```sh
pip install "./src[db]"
```

Python 3.12. The `db` extra (ChromaDB) is needed for `db build`, `db query` and `db list`, everything else runs without it.

Model conversion (building objects, human previews, importing object datasets) runs in Blender, which only ships for Python 3.11. It lives in its own package, [`blender/`](blender), and runs as a worker process:

```sh
uv tool install --python 3.11 ./blender
```

That puts `arena-assets-blender` on PATH. To use an existing environment instead, set `ARENA_ASSETS_BLENDER` to the worker command, for example `ARENA_ASSETS_BLENDER="/path/to/venv/bin/python -m arena_assets_blender"`.

The repository is also an `ament_python` ROS package: in the Arena workspace it is a member of the shared uv workspace and `colcon build` installs the `arena_assets` console script for `ros2 run arena_assets arena_assets`. The Blender worker is not part of that build.

## Usage

### Fetch prebuilt assets

```sh
# list available assets
arena-assets net default list

# download assets into a local directory
arena-assets net default fetch -o ./models Common/Material/ABS_Hard_Leather

# only download specific model formats, skip annotations
arena-assets net default fetch -o ./models --format usdz --no-annotation
```

`default` resolves to the public GCS bucket; any other bucket name can be passed instead.

### Build a database

```sh
arena-assets db ./models build -i ./dataset
```

Build options are passed with `-o key=value`:

| option | asset types | effect |
| --- | --- | --- |
| `formats` | object | comma-separated model formats to export (`*` for all) |
| `previews` | object | render preview images |
| `max-texture` | object, material | fit every texture into this many pixels on its longer side and store 16-bit maps with 8 bits per channel (default `2048`, `0` keeps them as they are) |
| `procthor` | object, material | export a ProcTHOR-compatible `asset-database.json` |
| `bake-mdl` | object | bake embedded MDL materials with Isaac Sim |
| `recompute-affordances` | object | recompute the z range of every affordance slot, hand-set ones included |

Each type is discovered under its own directory of the dataset (`Object`, `Material`, `Human`, `Sound`).

`--overwrite` controls re-building: `skip` (default) keeps existing entries, `overwrite` rebuilds them, `annotations` only refreshes annotations.

### Object lights

An object's `annotation.yaml` may carry `lights:`, one entry per light it emits: `name` (default `light`), `fixture` (`bulb`, `spot`, `downlight`, `panel` or `tube`), `pose` in the frame of the built model as `[[x, y, z], [w, x, y, z]]` or `[[x, y, z], [roll, pitch, yaw]]` (the light emits along the local -z axis of that pose, so `[1, 0, 0, 0]` shines straight down), `lumens`, `cct_K` (default 6500) and, for a spot, `cone_deg`. The build fills it from the lights of the source model (point lights become bulbs, spots keep their cone, area lights become panels or downlights, suns are skipped, 683 lm per Blender watt, color temperature from the light color) unless the source annotation already lists lights, and strips the lights from every exported model, so the annotation is the only place an object's light lives. glTF sources carry point and spot lights, USD sources also carry area lights.

A light may also carry `glow`, the name of the Blender material whose surfaces glow while the light is on. The build gives each extracted light the nearest emissive material: one whose Principled BSDF has a constant, non-black `Emission Color` and an `Emission Strength` above 0 (a textured emission never counts), measured from the light to the center of the bounding box of the faces using the material. Several lights may share one material, and an emissive material nearest to no light is left as it is. A hand-annotated `glow` must name a material used by the model, otherwise the build of that object fails with the list of its materials. Glow materials ship dark: their emission is cleared in every exported format, and the simulator lights them with the lamp. In the SDF export each glow material is a visual of its own, `glow_<material>` with the mesh `<name>.glow_<i>.dae` (`i` counts the glow materials in sorted name order), next to `visual` with the remaining faces in `<name>.dae`. A glow visual carries a `<material>` with the base color as `ambient` and `diffuse` and a black `specular` and `emissive`, unless the base color is textured.

### Object affordances

An object's `annotation.yaml` may carry `affordances:`, a map from an affordance kind to its slots. The only kind is `sit`, and an unknown kind fails the build of that object. A slot is a `bbox` over the seat surface in the frame of the built model, `[[x_min, x_max], [y_min, y_max]]`, and a `yaw`: the directions a person sitting there may face, an angle in radians, an axis (`+x`, `+y`, `-x`, `-y` or `xy`, as for `face`) or a list of them. A body rests anywhere inside the bbox and the legs hang over its edge, so each yaw points at an open edge of the seat. Slots of one object may overlap. Slots parse with `arena_simulation_setup.utils.affordance`, which the Arena workspace provides, so a standalone install reads and builds objects with affordances only next to it and lists such objects without their affordances.

```yaml
affordances:
  sit:
    - {bbox: [[-0.46, 0.48], [0.03, 0.72]], yaw: -y}
    - {bbox: [[-1.30, -0.56], [-1.13, -0.26]], yaw: [+x, -y]}
    - {}
```

The build adds a z range to each slot, from the floor of the object's bounding box up to the height of the seat surface seen from above inside the bbox: on a 2 cm grid of the model's top view, the 2 cm height bin between 0.2 and 0.9 m that covers the most cells, as the median of the cells within 5 cm of it. Without such a surface it takes `min(0.45, object height)` and warns. A slot whose source bbox already has a z range keeps it unless the build runs with `-o recompute-affordances`. A slot without a bbox, `{}` or `{yaw: ...}`, gets a square of up to 0.45 m clipped to the object, centered on the seat surface covering the most of the top view (the bbox center when there is none), facing its `yaw` or else `face`. At most one slot per kind may leave its bbox to the build. `--overwrite annotations` copies the z ranges of the last build, matching slots by their x and y ranges and the slot without a bbox to the one built slot no source bbox matches, when it faces the same way, and places the remaining slots (all of them with `-o recompute-affordances`) on the model the last build left in the database, so no model is exported again. Without that model, or when placing fails, the slots stay unplaced and the build warns.

`ObjectAnnotation.query_affordance(kind, slot)` returns a placed slot, one with a bbox and a yaw, as an `AffordanceSlot` and `affordance_slots()` returns every placed one. The slot schema (`Slot`, `structure_affordances`), the placed `AffordanceSlot` and the claims on them all live in `arena_simulation_setup.utils.affordance`, so the build and the runtime read slots the same way: an `AffordanceClaims`, one per placed object, keeps no two overlapping affordance boxes occupied at the same time, across all kinds. `available(kind)` lists the slots that overlap no occupied one, `closest(xy, kind)` returns the available slot nearest to an object-frame point (by distance to its box, 0 inside), and `claim` refuses a slot whose box overlaps an occupied one. Build it as `AffordanceClaims(annotation.affordance_slots())`, or as `AffordanceClaims.parse(affordances)` from the `affordances` block of a built `annotation.yaml`. A slot without a z range overlaps through all heights.

`arena-assets semantic seats` proposes sit slots with Gemini (install `arena_assets[vlm]` and set `GEMINI_API_KEY`):

```sh
arena-assets semantic seats propose ./dataset --renders ./models --out seats.yaml [--only Residential/Object/Sofa_B]...
arena-assets semantic seats apply seats.yaml ./dataset [--overwrite]
```

`propose` first asks which objects are for sitting, by asset path and description in batches of 50, unless `--only` names the assets. For each of them it sends the top-down and perspective previews of a build with `-o previews` and the worked example [seats_example.yaml](src/arena_assets/semantic/seats_example.yaml), asks `--samples` times (default 3) and keeps the slot count most answers agree on, the median box of matched slots and the facings most answers name. The example object is asked without its own example. It prints the tokens used and merges each object's proposal into `--out` as soon as it has it, keeping the objects already there. It stops at the first object whose samples all fail, the proposals so far kept. `apply` writes reviewed proposals into the source annotations, skipping objects that already have `sit` slots unless `--overwrite`, with which an empty list removes them. It checks every proposal before writing any file.

### Object articulation

An object's `annotation.yaml` may carry `articulation:`, the links of its meshes, the joints between them and named joint states. `links` maps each link name to the source nodes whose meshes it takes, and a link may list no node (the intermediate link of a joint with two degrees of freedom). A joint has a `name`, a `type` (`revolute`, `prismatic`, `continuous` or `fixed`), a `parent` and a `child` link, an `origin` point and a unit `axis`, a `limit` `[lower, upper]` for revolute (radians) and prismatic (meters) joints and an optional `velocity` (rad/s or m/s). `states` maps a state name to the values of some joints. The links form one tree under the root link, the one no joint has as its child.

```yaml
articulation:
  links:
    cabinet: [body]
    door: [door]
    drawer: [drawer_front, drawer_box]
  joints:
    - {name: hinge, type: revolute, parent: cabinet, child: door, origin: [0.45, -0.3, 0.0], axis: [0, 0, 1], limit: [0.0, 1.6]}
    - {name: slide, type: prismatic, parent: cabinet, child: drawer, axis: [0, -1, 0], limit: [0.0, 0.35], velocity: 0.2}
  states:
    closed: {hinge: 0.0, slide: 0.0}
    open: {hinge: 1.6, slide: 0.35}
```

The modeled mesh is the object at all joint values 0, where every link frame coincides with the model frame. In a build dataset, `origin` and `axis` are in the frame of the source model in meters, Z up, as for `collider:`, and the build moves them with the model when it stands the model upright on the floor, so the built annotation holds them in the frame of the built model. The articulation parses with `arena_simulation_setup.utils.articulation`, which the Arena workspace provides, and an articulation that does not parse keeps the object out of the build with the reason.

A mesh object of the model belongs to the link whose nodes name it or its nearest listed ancestor, both compared without Blender's duplicate suffix such as `.001`, and the root link takes every mesh no link claims. A USD source of an articulated object loads with every Xform kept as an object of its own, so an Xform holding a single mesh is a node under its own name. A listed node that matches no object fails the build of that object with the missing nodes and the objects the model has.

Next to the whole models at rest, which stay as without an articulation, the build writes:

- `links/<link>/` for every link with meshes: a static object bundle in the build's formats plus `obj` (`<link>.usdz`, `<link>.sdf/<link>.sdf`, `<link>.obj/<link>.obj`, ...), its geometry in the frame of the built model, and an `annotation.yaml` with the link's `name`, `path` and `bounding_box`. Its USD meshes get the static triangle-mesh collision the object's have, and an object `collider:` is not baked into links.
- `<name>.urdf`: one URDF link per articulation link, each link frame at its parent joint, visual and collision meshes `links/<link>/<link>.obj/<link>.obj` offset back into the model frame, a 1 kg inertia of the link's box, and a `<limit>` with the joint's velocity on revolute and prismatic joints.
- `<name>.articulated.usda`: a Z-up meters stage whose default prim `/<name>` is the articulation root, one rigid body Xform of 1 kg per link referencing the link's USD model (links without meshes are empty Xforms), a fixed joint from the world to the root link, and per joint a `PhysicsRevoluteJoint` (also for continuous joints, without limits), `PhysicsPrismaticJoint` or `PhysicsFixedJoint` between the parent and child link with the joint axis as its X axis, limits in degrees or meters. The build writes it when it exports a USD format.
- `swept` in the built `articulation:`, the box `[[x_min, x_max], [y_min, y_max], [z_min, z_max]]` the links cover over every reachable joint value. `bounding_box` stays the box at rest.

The database stores `articulated` for every object and the articulation as JSON for an articulated one. `--overwrite annotations` keeps the articulation of the last build, since a changed articulation changes the exported links, and warns that a source articulation the last build lacks needs a model build.

### Sound assets

A sound asset is a directory `Sound/<Name>/` holding its wav files and a `<Name>.yaml` manifest (version 2: `kind`, `level_db`, `loop`, `variants`, optional `desc` and `tags`). The manifest is the only annotation source: the build derives `annotation.yaml` from it (tags are the manifest tags, every variant's tags, and `kind::<kind>`, stamped with the ASA revision) and fails on a manifest with the wrong version or without `kind`, `level_db` or `variants`.

```sh
arena-assets db ./sounds build -t sound -i ./dataset
arena-assets db ./sounds query sound "robot motor hum" --filter "level_db>50"
arena-assets db ./sounds query sound "steps on wood" --filter "kind=footstep" -n 3
```

`kind`, `loop`, `level_db`, and the comma-joined variant ids are stored as metadata.

### Import upstream datasets

```sh
arena-assets import                         # registered sources by kind, with license verdict and host
arena-assets import object                  # sources of one kind
arena-assets import object gso              # item ids of one source
arena-assets import object gso Cole_Hardware_Hammer_Black -o ./gso
```

Each import converts items of an upstream dataset (objects, materials, worlds, scenarios, benchmark suites, sounds, recorded episodes) into local items under the imported root, marked with an `ATTRIBUTION.yaml`. [IMPORTING.md](IMPORTING.md) documents the pipeline and every source.

### Upload a built database

```sh
arena-assets net my-bucket author ./models -d Common
```

Files already present in the bucket with the same size are skipped. Uploading requires write access to the bucket: a token is taken from `--token`, the `GCS_ACCESS_TOKEN` environment variable, or `gcloud auth print-access-token`, in that order.

### Query

```sh
# best match (prints the asset path)
arena-assets db ./models query material "light brown wood"

# top 5 matches with distance scores
arena-assets db ./models query object "office chair" -n 5 --scores

# constrain by size (width/depth/height/volume in meters, repeatable)
arena-assets db ./models query object "office chair" --filter "height<1.0" --filter "volume>0.05"

# list everything of a type
arena-assets db ./models list material
```

All logs and progress bars go to stderr; stdout only carries the results, so output can be piped.

### Splat walls and objects

`arena-assets splat` turns a trained 3D Gaussian splat of a real wall into an object bundle that a wall kind places with its `model:` directive. With `--object` it frames a free-standing object instead, see [Splat objects](#splat-objects). The recipe for a wall, from capture to a pushed bundle:

1. Capture a phone video or photos of the wall area and run COLMAP on them (3DGRUT reads COLMAP layouts).
2. Train with 3DGRUT and export OpenUSD:

   ```sh
   python train.py --config-name apps/colmap_3dgut_mcmc.yaml path=<scene> out_dir=<out> experiment_name=<name> \
       dataset.downsample_factor=2 strategy.add.max_n_gaussians=4000000 \
       export_usd.enabled=true export_usd.export_cameras=false export_usd.export_background=false
   ```

   4M Gaussians at downsample factor 2 gave the best look, about 20 min on an RTX PRO 6000. Export without cameras and background, because Isaac's SpawnPrims turns the default light off for assets that carry lights.
3. Register and crop the export into a bundle directory that the object resolver finds, for example a world's `assets/Common/Object/<name>`:

   ```sh
   arena-assets splat <out>/<name>/<run>/export_last_lightfield.usdz ./assets/Common/Object/<name> --points <name>.txt --width 3.2 --height 2.5
   ```

   Registration, crop and `--drop` boxes are given in the wall frame: meters, origin at the base midpoint of the captured wall, X along the wall, Y facing into the room and Z up, with Y = 0 on the captured wall plane. `--points FILE` fits the registration (uniform scale, rotation, translation) from at least 3 non-collinear correspondences, one `sx sy sz tx ty tz` per line (a point in the source stage and the same point in the wall frame), and prints the fitted scale and the RMS residual in meters. `--matrix` takes a known transform instead: 16 numbers in one quoted argument, row-major, `p_target = M @ p_stage`.

   The crop keeps X within `--width`, Z from `--floor` (default 0.03) to `--height`, and Y from `-behind` (default 0) to `--depth` (default 0.05). Captured geometry is seen by Isaac lidar and depth cameras but has no collision, so the default keeps only 5 cm in front of the wall plane and deeper crops are opt-in. `--drop 'XMIN YMIN ZMIN XMAX YMAX ZMAX'` (repeatable) removes the Gaussians inside a wall-frame box. `--max-scale` (default 0.2 m) removes oversized Gaussians, which would blow up the extent.

   The bundle origin is the wall base midpoint at the back of the crop: the bundle is the wall frame shifted by `+behind` along Y, and the physical wall of the wall kind sits at its Y = 0, so the whole capture lies in front of that wall. With the default `--behind 0` nothing behind the captured wall plane is kept. With `--behind B` the captured wall surface stands B meters in front of the physical wall, seen by lidar and depth cameras but not solid.

   The bundle holds `<name>.gaussians.usdc` (the kept Gaussians), `<name>.usda` (a Z-up wrapper in meters that carries the registration), `annotation.yaml` and, with `--collider-mesh`, the collider (see [Splat colliders](#splat-colliders)).
4. The rule: anything standing out from the wall must be wholly inside or wholly outside the crop, otherwise its cut parts float. A floor plant whose leaves reached into the crop had to be taken out with `--drop`.
5. Publish with `arena asset push object <name>`.
6. Place it in a wall kind:

   ```yaml
   main:
     - material: Plaster_Wall      # the wall: collision, lidar, map, and the backdrop behind the capture
       height: 2.6
       shadows: false
     - model: <name>
       at: 50%
   ```

   - The segment is an ordinary drawn wall on the wall line.
   - It shows through holes in the capture.
   - `shadows: false` keeps simulator shadows off the capture, whose lighting is baked in.

   Isaac only: Gazebo and the other simulators skip the USD-only object and draw the wall.

#### Splat objects

`--object` frames a free-standing object, captured from all sides, as a bundle that worlds and scenarios place like any other static object. Capture and training follow steps 1 and 2 above, then:

```sh
arena-assets splat <export>.usdz ./assets/Common/Object/<name> --object --matrix '<16 numbers>' --width 1.2 --depth 1.2 --height 1.1
```

- Registration, crop and `--drop` boxes are given in the object frame: meters, origin on the floor under the object center, X and Y level with Y toward the object front, Z up. `--points` and `--matrix` work as for walls.
- The crop keeps X within `--width`, Y within `--depth` (required), both centered on the origin, and Z from `--floor` (default 0.03, which cuts the captured floor away) to `--height`. `--behind` applies to walls only.
- The bundle origin is the object frame origin, so a pose at floor height stands the object on the floor, and `annotation.yaml` records the crop as the bounding box.

Place it like any other object, in a world level's or a scenario's `objects:` list:

```yaml
objects:
  - {name: garden_table, model: <name>, pose: [5.0, 4.0, 0.0]}
```

Without a collider the object is visual only. Isaac cameras, lidar and depth cameras see it, physics does not: robots drive through it. The task generator keeps spawns and goals off its bounding box, and humansim receives the box as an obstacle. The capture's lighting is baked in. Isaac only: Gazebo and the other simulators skip it.

#### Splat colliders

`--collider-mesh FILE` gives a splat bundle a collider at build time. FILE is a triangle mesh of the same capture, `.obj` or `.ply`, in the source stage frame of the export, for example a surface reconstructed from the training views. Building that mesh is outside this tool. It is registered with the same `--matrix` or `--points` as the Gaussians and cropped by the same box: a triangle stays only with all its corners inside the crop and none inside a `--drop` box, and the verb prints how many it keeps. The kept triangles are baked into `<name>.usda` as described under [Colliders](#colliders). The bundle has no SDF of its own, so the verb also writes `<name>.sdf/<name>.sdf` holding only the collision: Gazebo then has the obstacle without the capture.

### Colliders

An object's `annotation.yaml` may name a collider mesh under `collider:`. It replaces the object's collision in every model the bundle holds, while cameras, lidar and depth cameras keep seeing the visual model:

- USD (`<name>.usdz`, `.usd`, `.usda` or `.usdc`): an invisible static triangle-mesh collider `collider/mesh` under the default prim, with the collision of the visual meshes switched off (`physics:collisionEnabled = false`). Without one, the build gives every visual mesh a static triangle-mesh collision, as the SDF has.
- SDF (`<name>.sdf/<name>.sdf`): the only `<collision>`, named `collider`, with the mesh `<name>.collider.obj` next to the SDF. An SDF without one collides with its visual meshes.
- The mesh itself stays in the bundle as `<name>.collider.obj`, which the built annotation names.

In a build dataset, `collider:` names an `.obj` or `.ply` triangle mesh in the object's source directory, in the frame of the source model in meters, Z up. The build moves the mesh with the model when it stands the model upright on the floor, then bakes it into every exported format.

`arena-assets collider BUNDLE MESH` does the same for a built bundle directory, for example a pulled object or a world-local copy:

```sh
arena-assets collider ./assets/Common/Object/<name> <hull>.obj
arena-assets collider ./assets/Common/Object/<name> --remove
```

- MESH is an `.obj` or `.ply` triangle mesh (ASCII or binary, polygons are split into triangles) in the bundle frame, the frame of the bundle's models in meters, Z up. `--matrix` maps a mesh from another frame into it.
- It bakes into the models present in the directory and prints them. A bundle pulled with only some formats gets the collider only in those.
- Running it again replaces the collider. `--remove` takes it out of the models, which then collide as they were built, and deletes `<name>.collider.obj` and the annotation entry.
- `arena asset push object <name>` publishes the changed models with the bundle.

## If your materials are missing in the output model

USD files can contain embedded MDL materials, which need to be baked before Blender can bake them correctly.
You can specify `-o bake-mdl=/path/to/isaacsim/python.sh` to enable automatic baking of MDL materials during database build.
Specifying `-o bake-mdl` without a path uses a docker image from the [Nvidia nvcr registry](https://nvcr.io/nvidia/isaac-sim). (May not work consistently.)

## Development

```sh
pip install "./src[db,test]"
pre-commit install

pytest tests/
ruff check . && ruff format src
```
