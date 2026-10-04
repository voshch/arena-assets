import os

from arena_assets.impl.build import DatabaseBuilder
from arena_assets.utils.Database import Database

from . import DATABASE_NAME, Annotation, AssetType


def list_database(database_path: str, asset_type: AssetType) -> list[Annotation]:
    db = Database(os.path.join(database_path, DATABASE_NAME))
    result = db.list_all(asset_type.value)

    annotation_t = DatabaseBuilder.Builder(asset_type)._annotation_cls

    return [annotation_t.from_metadata(dict(item)) for item in result["metadatas"] if item]
