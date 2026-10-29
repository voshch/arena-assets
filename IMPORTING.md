# Importing upstream datasets

`arena asset import` converts items of public datasets into Arena items: objects, materials, worlds, scenarios, benchmark suites, sounds and recorded episodes. An imported item lands in the imported root and resolves like a local one, with an `ATTRIBUTION.yaml` naming where it came from. The registry in [importers/__init__.py](src/arena_assets/importers/__init__.py) is authoritative for the sources, their kinds and their license verdicts.

## Usage

```sh
arena asset import                                   # registered sources by kind, with license verdict and host
arena asset import object                            # sources of one kind
arena asset import object gso                        # item ids of one source
arena asset import object gso Cole_Hardware_Hammer_Black
arena asset import world procthor val/0 val/1        # several items
arena asset import world barn --all                  # every item of a source
arena asset import object gso <item> --out ./gso     # another target root
```

Over MCP, the `assets` domain offers the same: `assets_import_sources` lists the datasets, `assets_import_items` one dataset's item ids (with a glob and a limit), and `assets_import` converts chosen items as a job whose result lists the item paths. `--json` prints either listing as JSON.

`arena asset import` is the Arena CLI front of `arena-assets import KIND SOURCE ITEM... [-o OUT]`, which works the same outside Arena. Tab completion of `arena asset import KIND SOURCE` offers the source's item ids.

## How an import works

Importers are registered by kind and source, so one source can serve several kinds (`import world barn 17` is one challenge world, `import suite barn barn` the whole benchmark). Each item is fetched into the source's cache under `$ARENA_ASSETS_DIR/.import/SOURCE`, converted into the kind's directory layout under the target root and marked with an `ATTRIBUTION.yaml` (`source`, `url`, `license`, `authors`, `citation`).

The default target is the imported root, `$ARENA_ASSETS_DIR_IMPORTED`, else `$ARENA_ASSETS_DIR/_imported`, with worlds and the scenarios that bring their world under its `worlds/`. The resolvers of every kind read it after the curated local root, so an imported item resolves like any local one.

A bare `import KIND SOURCE` caches the source's item list in `$ARENA_ASSETS_DIR/.import/SOURCE/items.KIND.json`, for 24 h when the host's listing can change (Fuel, Dataverse) and for good when the source is pinned to a revision, and shows the old list when the host fails. Tab completion reads the same file and starts the listing in the background when the file is missing or expired. With `--all`, an item that fails is logged with its reason and skipped, and the import ends with a count and exit code 1 when any failed. An explicit item that fails stops the import with a one-line error naming it. A download that gets a 5xx, 408 or 429 answer, no answer within 30 s, a dropped connection or a short body is tried four times with a doubling pause before the import fails.

The license verdict of a source is `yes` (usable as is, attribution where the license asks for it), `nc` (non-commercial), `sa` (share-alike), `no`, or `unclear` (the upstream terms are missing or rest on data under unknown terms). A source that takes only some licenses refuses an item under any other.

Object sources are staged as a dataset and run through the same `arena-assets db build` as native objects (`formats=usdz,sdf,fbx` plus `previews`), so they need the [Blender worker](README.md#installation) and the `db` extra, and `OUT/Common` is a queryable database. Object and material builds fit every texture into 2048 px on its longer side (the `max-texture` build option). Material sources need no Blender.

## Objects

Every object source writes `Common/Object/<item>` with the `usdz`, `sdf` and `fbx` models, previews, an `annotation.yaml` tagged `origin::<source>` and its attribution.

The articulated sources ([simready](#simready), [artvip](#artvip), [partnet_mobility](#partnet_mobility), [infinigen](#infinigen) and [artiverse](#artiverse)) also write the joints of an item as the `articulation:` block of its annotation, which the build turns into link bundles, a URDF and an articulated USD (see [Object articulation](README.md#object-articulation)). The block is in the frame the staged model loads in, meters and Z up, with the modeled pose at every joint value 0. Every upstream joint is kept, casters, wheels and fixed joints included, a joint whose name would take a field name the runtime reserves (such as `angle` or `state`) getting a `_2` suffix: an unlimited revolute joint becomes `continuous`, a joint between two parts runs from the root part outward whichever way upstream wrote it, and a rigid part no joint reaches hangs on the root link by a `fixed` joint. A joint that only anchors a part to the world is dropped, and a joint that would close a loop is dropped with a warning. Doors, drawers, lids, covers, hatches, flaps, windows and hinged parts (not their handles, knobs or buttons) get the states `closed` at the modeled pose and `open` at the limit end farthest from it, unless the source names its own states. A joint's `role` comes from its upstream joint and part names, the first of the handle-like parts, the named openings and then a bare hinge they show. USD sources keep the physics they author, converted to SI units at the joint's values: PhysX joint friction and armature, an axis friction effort, a drive's damping, a drive's stiffness as a return `spring` when the drive targets the modeled pose (a drive aimed elsewhere, such as ArtVIP's drives toward the open pose, is the source's actuator and keeps only its damping), and per link the physics material most of its colliders are bound to. URDF sources keep `<dynamics>` damping and friction. A limit that excludes the modeled pose, as some upstream joints have, is made to reach it with a warning, so every part can rest closed: a revolute range half a turn off (such as ArtVIP's `[-pi, -1.22]` on a door modeled closed) is turned back by half a turn, keeping its span and its opening direction, and any other is widened to 0. An item without a moving joint stays a plain object.

### gso

[Google Scanned Objects](https://app.gazebosim.org/GoogleResearch/fuel/models) from Gazebo Fuel, about 1k scanned household objects under CC BY 4.0. The item id is the Fuel model name (`import object gso Cole_Hardware_Hammer_Black`). An item stages the one mesh the model's SDF shows with the scale, pose and albedo map the SDF gives it, rewritten into the axes the builder's importer expects. The annotation takes the semantic fixes of [semantic](src/arena_assets/semantic), and the item list is fetched again after 24 h.

### ai2thor

[AI2-THOR](https://ai2thor.allenai.org/) objects, the ProcTHOR, iTHOR and RoboTHOR asset set under CC BY 4.0, read from the [ManiSkill copy](https://huggingface.co/datasets/haosulab/AI2THOR) of the HSSD glTF conversion at a pinned revision. The item id is the AI2-THOR asset name (`import object ai2thor CreditCard_3`). Only the bytes of the requested glTF files are read from the remote archive, and the object type comes from the ProcTHOR asset database. The list holds numbered variants of one object (`Fridge_1`, `Fridge_10`). The [procthor](#procthor) world source imports its furniture through this source.

### openrobotics

[Open Robotics](https://app.gazebosim.org/OpenRobotics/fuel/models) furniture, appliance, kitchen, hospital, office and warehouse models on Gazebo Fuel, only those under CC0 or CC BY 4.0. The item id is the Fuel name with underscores for every run of other characters (`import object openrobotics Dining_Chair`). An item is staged like a [gso](#gso) one and records the model's own license and authors. A model under another license, one that is not a single OBJ or COLLADA mesh, and one the builder would load empty or rescale (a footprint over 5 m) are refused. The item list is fetched again after 24 h.

### simready

[NVIDIA SimReady assets](https://huggingface.co/datasets/nvidia/simready-assets) on Hugging Face at a pinned revision, about 2.5k household, hospital, warehouse, factory and vehicle part USD models with physics payloads under CC BY 4.0. The item id is the asset folder (`import object simready bed_hospital_standard_a01`). An item fetches its package's metadata and USD layers, checks the CC BY 4.0 license the package states, then fetches only the textures and materials its stage refers to. A footprint over 5 m is refused. The root USD is built with its scene instances realized. The built item keeps the package's SimReady metadata and its PhysX, MuJoCo and Newton physics payloads under `simready/`, which the model formats do not carry, and the download is removed once the item is built. The UsdPhysics joints of the composed stage become the articulation: each rigid body a link named after its prim without the `_obj_NN` suffix (`wheelPin_01_obj_00` is `wheel_pin_01`), each joint named after its prim without the `joint_` and `_joint_NN` affixes. `bed_hospital_standard_a01` keeps its four caster swivels and four wheels as `continuous` joints, its brake as a `revolute` joint and its glass as a `fixed` one. Most SimReady assets have no joint and stay plain objects.

### simready_dsx

[NVIDIA SimReady AI factory assets](https://huggingface.co/datasets/nvidia/simready-dsx) on Hugging Face at a pinned revision, today the GB300 rack with its physics, electrical and thermal metadata, under CC BY 4.0 (`import object simready_dsx GB300`). Imported like a [simready](#simready) item.

### artvip

[ArtVIP](https://huggingface.co/datasets/X-Humanoid/ArtVIP) on Hugging Face at a pinned revision, the 439 articulated household, furniture, appliance, lab, medical and industrial objects of its `Articulated_objects` folder as USD stages with UsdPhysics joints, under Apache 2.0. The item id is the object folder name in snake case, which the built object keeps (`import object artvip eket_cabinet_with_door_brown_walnut_effect_35x35x35cm`). An item fetches the folder's model stage, then every layer and texture the composed stage refers to, leaving out the thumbnails and the `origin/` base variants. A footprint over 5 m is refused. Each rigid body becomes a link named after its prim without the `E_` prefix, and the category folder an object sits in becomes its kind tag. The joint drives and the magnetic, damping and trigger behavior of the modular interactive objects are not carried, and the interactive and digital twin scenes are not imported. The download is removed once the item is built.

### partnet_mobility

[PartNet-Mobility](https://huggingface.co/datasets/sapien-sim/PartNetMobility), the SAPIEN articulated object set, on Hugging Face at a pinned revision behind a gate: a token with access goes in `HF_TOKEN` (or the file `HF_TOKEN_PATH` or `$HF_HOME/token` names), and an item fetched without access ends with a one-line error naming the page that grants it. The listing works without access. About 2.3k URDF models of 46 categories with part meshes and textures. The item id is the model id and the object `partnet_mobility_<id>` (`import object partnet_mobility 100013`). An item merges the OBJ meshes of every URDF link into one OBJ whose objects carry the link names, each placed by its link and visual origin, with each mesh file's materials renamed apart, their reflective `illum 3` (which Blender loads as fully metallic) made plain `illum 2`, and the textures kept, and turns the URDF joints into the articulation. `meta.json` gives the description and kind tag, and the labels of `semantics.txt` pick the doors and drawers for the `closed` and `open` states, `closed` at the lower limit and `open` at the upper one, since PartNet-Mobility models some parts ajar with the closed pose at the lower limit. Floating and planar joints are kept as fixed ones. The models are size-normalized rather than metric, so an item is scaled uniformly to the typical size of its category (the `SIZES` table in [partnet_mobility.py](src/arena_assets/importers/partnet_mobility.py)): the height for floor-standing things such as doors (2.05 m), refrigerators (1.75 m) and tables (0.75 m), the shorter horizontal side for storage furniture (0.5 m), and the longest side for small things such as microwaves (0.5 m) and remotes (0.2 m) and for ovens (0.8 m), a category that holds flat cooktops as well as ranges. A table's longest side is capped at 2 m, so a long low coffee table is not stretched to dining height. A category outside the table keeps the upstream scale, read as meters. The verdict is `nc`: the terms allow non-commercial research and education only, and the shapes come from ShapeNet.

### infinigen

[Infinigen-Articulated](https://github.com/princeton-vl/infinigen) sim-ready assets under BSD 3-Clause, generated on demand by a local Infinigen install. The item id is `<category>/<seed>` and the object `infinigen_<category>_<seed>` (`import object infinigen door/1001`, `import object infinigen dishwasher/1001`). The listing offers seeds 1001 to 1010 of every category of Infinigen's `OBJECT_CLASS_MAP`, and any integer seed imports. An item runs Infinigen's `spawn_simready` with its USD exporter in the interpreter `ARENA_ASSETS_INFINIGEN` names, else in the `python3` on PATH when that imports Infinigen, and keeps the export in the cache. Without an install the import ends with the command that makes one:

```sh
git clone https://github.com/princeton-vl/infinigen.git && cd infinigen && uv venv --python 3.11 && INFINIGEN_MINIMAL_INSTALL=True uv pip install -e '.[sim]' && export ARENA_ASSETS_INFINIGEN=$PWD/.venv/bin/python
```

Infinigen's USD export writes one face size per face-vertex index, which Blender cannot load, and the staging gives each mesh one face per run of indices instead. The fixed `world` body becomes the link `base`, the other bodies keep their `link_<n>` names and the joints their labeled names (`door_hinge_left_0`).

### artiverse

[Artiverse](https://huggingface.co/datasets/3dlg-hcvc/artiverse) from Simon Fraser University on Hugging Face at a pinned revision behind a gate with manual approval: without access the import ends with a one-line error naming the page to request it at. The dataset is 5.4k part-segmented glTF models of 88 categories in layered releases, a base and an extension that add model folders and corrections that overlay files on them, about 120 GB of tar.gz chunks in all. The listing reads `releases.json` and each release's manifest (about 1 MB) once into the cache and names every model `<category>/<id>` (`import object artiverse armoire/2154c63d9146908de9f3e9c758fef992`, the bare id works too), the object is `artiverse_<id>` with the category as description and kind. A gzip stream has no index, so an import streams each chunk whose manifest holds the model, in release order, keeps the model's `<id>.segmented.glb` and `<id>.articulations.json`, a later release replacing an earlier file, and stops reading once the stream has passed the model's folder. A model near the end of a 38 GB base chunk thus takes minutes to reach, one near its start seconds, and nothing but the two files is stored. Each part, a child of the glTF root node with an `id` extra, becomes the link `<label>_<id>`, its node renamed to match, and hangs on its parent in the part graph. A cylindrical or screw joint becomes a prismatic and a revolute joint chained over a link without meshes (`<link>_slide`), a universal joint two revolute joints chained the same way (`<link>_axis1`), and a free part hangs on its parent by a fixed joint. Axes and origins are turned from the glTF's Y up into the Z-up model frame. The `closed` and `open` values the articulation file names in `motionStates` are laid over the `closed` and `open` rule, a value it gives winning, since the files name mostly `closed` alone. Its other named values become detents of their joint, skipping a name the runtime reserves. A screw joint loses the coupling of its turn and slide. The verdict is `nc`: each object stays under the terms of the dataset it came from (ShapeNet, ABO, 3D-FUTURE, HSSD, Objaverse and others).

## Materials

Material sources share one base, `MaterialImporter` in [material.py](src/arena_assets/importers/material.py): a source names the map files of an item (`Maps`: sRGB color, OpenGL normal, linear roughness, metalness and occlusion, or the last three packed as one occlusion-roughness-metalness file) and its description, tags, authors and page, and the base writes the `Common/Material/<name>` item every simulator reads. That item is `<name>.mdl`, an OmniPBR material binding the maps it has, beside `<name>/<name>_BaseColor`, `_N`, `_ORM`, `_R`, `_M` and `_AO` files copied as they came, with the normal map's V tangent unflipped so the OpenGL map is right in Isaac and Gazebo alike. The item goes through the same `db build` as a native material, so `OUT/Common` is a queryable database. A material name is an MDL identifier, a letter followed by letters, digits and underscores, and an item without a color map is refused.

### polyhaven

[Poly Haven](https://polyhaven.com/textures) textures under CC0 (`import material polyhaven aerial_asphalt_01`). An item takes the 2k JPEG color, OpenGL normal and packed `arm` maps of a texture, the separate roughness, metalness and occlusion maps where no packed one exists.

### ambientcg

[ambientCG](https://ambientcg.com/) materials under CC0 (`import material ambientcg Metal032`). An item takes the color, OpenGL normal, roughness, metalness and occlusion maps of the material's `2K-JPG` zip.

## Worlds

World sources write `worlds/<name>` in world format 2, which `WorldIdentifier` reads after the `ARENA_WORLD_PATH` roots and before the shipped worlds, so an imported world launches by name (`arena launch world:=procthor_val_0`).

### procthor

[ProcTHOR-10K](https://github.com/allenai/procthor-10k) houses under Apache 2.0, 10k train, 1k val and 1k test. The item id is `<split>/<index>` (`import world procthor val/0`) and the world `procthor_<split>_<index>`: zones named by room type, the walls cut around every door hole, openable doors as `hinged` doors, and the furniture imported through the [ai2thor](#ai2thor) object source and placed as static entities. See [worlds/README.md](../arena_simulation_setup/worlds/README.md#how-worldidentifier-resolves-a-name).

### houseexpo

[HouseExpo](https://github.com/TeaganLi/HouseExpo) floor plans at a pinned revision, 35,126 plans of labeled rooms and walls in one 25 MB archive, downloaded once and indexed in the cache. The item id is the plan id and the world `houseexpo_<plan id>`, with zones from the room boxes clipped to the floor. The repository is MIT licensed and its plans are built on SUNCG, so the verdict is `unclear`. See [worlds/README.md](../arena_simulation_setup/worlds/README.md#how-worldidentifier-resolves-a-name).

### barn

[BARN challenge](https://github.com/Daffan/the-barn-challenge) worlds 0 to 299 under MIT. The item id is the world number (`import world barn 17`) and the world `barn_<index>`, built by the `barn` world generator from the upstream world file, with the challenge start and goal as its scenario. The whole benchmark is the [barn suite](#barn-1).

## Scenarios

Scenario sources write recorded pedestrians as replay scenarios (`human:=replay`, see the [replay adapter](../task_generator/task_generator/simulators/human/README.md#replay-adapter-humanreplay)), each bringing the world it plays in where that is not imported yet. Tracks are resampled to 10 Hz with one `stays:` entry per continuous stay for a path follower.

### eth_ucy

The ETH and UCY pedestrian sequences in [SocNavBench's](https://github.com/CMU-TBD/SocNavBench) copies at a pinned revision. The item id is the sequence, `eth`, `hotel`, `univ`, `zara01` or `zara02` (`import scenario eth_ucy zara01`), and the world's floor is the area the pedestrians walked. zara01 and zara02 share one world. The trajectories carry no license, so the verdict is `unclear`.

### socnavbench

The [SocNavBench](https://github.com/CMU-TBD/SocNavBench) curated episodes on eth, zara01 and zara02, 17 of them, each a replay scenario with the robot start and goal as one `goto` phase in `socnavbench_eth` or `socnavbench_zara` (`import scenario socnavbench <episode>`). The 16 episodes on hotel, univ and the double hotel are refused. The episodes are MIT licensed and the trajectories carry no license, so the verdict is `unclear`. The whole benchmark is the [socnavbench suite](#socnavbench-1).

### juelich

The [Juelich pedestrian dynamics archive](https://ped.fz-juelich.de/da/doku.php) under CC BY 4.0: the BaSiGo 2013 bidirectional corridor and 90 degree crossing and the HERMES 2009 corridors, bottleneck and T-junction. The item id is `<set>/<run>` with set `basigo_bidirectional`, `basigo_crossing_90` or `hermes_<experiment>`, and the world `juelich_<set>_<k>` is one per distinct lab layout of a set, its pillars and inner walls walled in.

### madras

The [MADRAS](https://zenodo.org/records/13830435) TopView sequences of the Fete des Lumieres 2022 in Lyon under CC BY 4.0, dense crowds from one archive (`import scenario madras topview_1a`, up to `topview_2f`). They play in `madras_topview`, an open floor 2 m around every tracked point.

### glow

The [Eindhoven GLOW 2019](https://zenodo.org/records/7007358) crowd recording under CC BY 4.0, cut into 5-minute periods (`import scenario glow <YYYY-mm-ddTHHMM>`). The first import downloads and indexes the 1 GB recording once. The periods play in `glow`, an open floor 2 m around every tracked point.

## Benchmark suites

Suite sources write a suite bundle with its worlds, a contest and a report manifest into the imported root, where they resolve ahead of the shipped ones. See [configs/benchmark/README.md](../arena_evaluation/arena_evaluation/configs/benchmark/README.md) for the stages, contestant rules and metrics.

### barn

The [BARN challenge](https://github.com/Daffan/the-barn-challenge) as one item (`import suite barn barn`): `suites/barn` with one stage per world and the 300 worlds bundled, `contests/barn` with the challenge episode rules, and `manifests/barn`, which tabulates the `barn_score` the challenge's `run.py` computes. Under MIT. See [BARN](../arena_evaluation/arena_evaluation/configs/benchmark/README.md#barn).

### socnavbench

[SocNavBench](https://github.com/CMU-TBD/SocNavBench) as one item (`import suite socnavbench socnavbench`): `suites/socnavbench` with one jackal stage per curated [socnavbench episode](#socnavbench) and its time budget, the two scene worlds bundled, `contests/socnavbench` with SocNavBench's 0.3 m goal margin, and `manifests/socnavbench`. See [SocNavBench](../arena_evaluation/arena_evaluation/configs/benchmark/README.md#socnavbench). The verdict is `unclear`, as for the scenarios.

## Sounds

Sound sources stage each clip as a manifest beside its unchanged wav and build it with `arena-assets db build -t sound` as `Common/Sound/<source>_<item>`. See [configs/sounds/README.md](../arena_simulation_setup/configs/sounds/README.md#imported-sounds) for the manifest values.

### fsd50k

[FSD50K](https://huggingface.co/datasets/Fhrozen/FSD50k) clips under CC0 or CC BY 3.0, only those whose AudioSet labels map to a sound kind with a default asset through [fsd50k_kinds.yaml](../arena_simulation_setup/configs/sounds/fsd50k_kinds.yaml). The item id is the Freesound id (`import sound fsd50k <freesound id>`), and the uploader and Freesound URL go into the attribution.

### cmu_arctic

[CMU ARCTIC](http://festvox.org/cmu_arctic/) read speech of the seven speakers awb, bdl, clb, jmk, ksp, rms and slt, about 1,130 utterances each, under the CMU ARCTIC license. The item id is `<voice>_<utterance>` (`import sound cmu_arctic bdl_a0001`), a sound of kind `speech` with the prompt text as `desc` and tags for the voice, gender and accent.

## Recorded episodes

### musohu

[MuSoHu](https://dataverse.orc.gmu.edu/) rosbag1 human navigation runs from the GMU Dataverse under CC0. The item id is the bag name (`import episode musohu <bag>`), and the run becomes `<bag>/episodes/episode_000`, a reference episode of the robot model `human` in the evaluation recorder's layout: scan, odom and cmd_vel on the recorder's topics, every other topic with a ROS 2 type under the robot namespace. A bag missing a needed topic is refused before download. The item list is fetched again after 24 h. See [ingestion/README.md](../arena_evaluation/arena_evaluation/arena_evaluation/ingestion/README.md#files).

## Adding a source

A source subclasses `Importer` (or `MaterialImporter`, or the sound base in `arena_simulation_setup.adapters.sound`), implements `items`, `fetch`, `convert` and `attribution`, and registers an `ImporterMeta` with its kind, source name, `module:Class` target, license verdict, host and description in [importers/__init__.py](src/arena_assets/importers/__init__.py). The module lives in the package that owns the kind it writes, and `default_out` names the root its resolvers read.
