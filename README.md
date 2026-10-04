# arena-assets

Build, query, and manage 3D model databases for [Arena](https://github.com/Arena-Rosnav).

`arena-assets` converts datasets of assets (objects, materials, humans, and sounds) into a searchable database: models are converted with Blender, optionally baked with Isaac Sim, and indexed in [ChromaDB](https://www.trychroma.com/) so they can be queried by natural-language description.

## Installation

```sh
pip install ./src
```

Note: if you want to install `arena_assets[build]`, you must use Python 3.11.* (bpy dependency).

The repository is also an `ament_cmake` ROS package; building it in a colcon workspace installs the CLI into a dedicated venv.

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
| `procthor` | object, material | export a ProcTHOR-compatible `asset-database.json` |
| `bake-mdl` | object | bake embedded MDL materials with Isaac Sim |

Each type is discovered under its own directory of the dataset (`Object`, `Material`, `Human`, `Sound`).

`--overwrite` controls re-building: `skip` (default) keeps existing entries, `overwrite` rebuilds them, `annotations` only refreshes annotations.

### Sound assets

A sound asset is a directory `Sound/<Name>/` holding its wav files and a `<Name>.yaml` manifest (version 2: `kind`, `level_db`, `loop`, `variants`, optional `desc` and `tags`). The manifest is the only annotation source: the build derives `annotation.yaml` from it (tags are the manifest tags, every variant's tags, and the kind) and fails on a manifest with the wrong version or without `kind`, `level_db` or `variants`.

```sh
arena-assets db ./sounds build -t sound -i ./dataset
arena-assets db ./sounds query sound "robot motor hum" --filter "level_db>50"
arena-assets db ./sounds query sound "steps on wood" --filter "kind=footstep" -n 3
```

`kind`, `loop`, `level_db`, and the comma-joined variant ids are stored as metadata.

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

## If your materials are missing in the output model

USD files can contain embedded MDL materials, which need to be baked before Blender can bake them correctly.
You can specify `-o bake-mdl=/path/to/isaacsim/python.sh` to enable automatic baking of MDL materials during database build.
Specifying `-o bake-mdl` without a path uses a docker image from the [Nvidia nvcr registry](https://nvcr.io/nvidia/isaac-sim). (May not work consistently.)

## Development

```sh
pip install "./src[test]"
pre-commit install

pytest tests/
ruff check . && ruff format src
```
