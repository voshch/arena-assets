import math
import os
import re

from arena_assets.impl.build import DatabaseBuilder
from arena_assets.utils.Database import Database
from arena_assets.utils.logging import get_logger

from . import DATABASE_NAME, Annotation, AssetType

logger = get_logger("query")

RRF_K = 60
SEMANTIC_POOL = 20


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _lexical_rank(query: str, corpus: list[Annotation]) -> list[Annotation]:
    """Rank corpus entries by name/tag token overlap with the query. Entries without overlap are dropped."""
    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    scored = []
    for annotation in corpus:
        name_tokens = _tokens(annotation.name) | _tokens(annotation.name_text)
        tag_tokens = _tokens(" ".join(annotation.tags))
        score = 0.0
        if name_tokens:
            score += len(query_tokens & name_tokens) / len(query_tokens | name_tokens)
            if query_tokens in (_tokens(annotation.name), _tokens(annotation.name_text)):
                score += 1.0
        if tag_tokens:
            score += 0.5 * len(query_tokens & tag_tokens) / len(query_tokens | tag_tokens)
        if score > 0:
            scored.append((score, annotation))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [annotation for _, annotation in scored]


def query_database(database_path: str, asset_type: AssetType, query_target: str, n: int = 1, where: dict | None = None) -> list[tuple[Annotation, float]]:
    logger.info("Querying database at %s for %s '%s'", database_path, asset_type.value, query_target)

    db = Database(os.path.join(database_path, DATABASE_NAME))
    annotation_t = DatabaseBuilder.Builder(asset_type)._annotation_cls

    result = db.query(asset_type.value, query_target, max(n, SEMANTIC_POOL), where=where)
    logger.debug("Query result: %s", result)

    metadatas = result["metadatas"]
    if not metadatas or not metadatas[0]:
        raise ValueError("No results found in the database.")

    distances = result["distances"]
    scores = distances[0] if distances else [0.0] * len(metadatas[0])

    semantic: dict[str, tuple[Annotation, float]] = {}
    for metadata, distance in zip(metadatas[0], scores, strict=True):
        annotation = annotation_t.from_metadata(dict(metadata))
        semantic[annotation.path] = (annotation, distance)

    corpus = [annotation_t.from_metadata(dict(metadata)) for metadata in db.list_all(asset_type.value, where=where)["metadatas"] if metadata]
    lexical = _lexical_rank(query_target, corpus)

    fused: dict[str, float] = {}
    for rank, path in enumerate(semantic):
        fused[path] = fused.get(path, 0.0) + 1.0 / (RRF_K + rank)
    for rank, annotation in enumerate(lexical):
        fused[annotation.path] = fused.get(annotation.path, 0.0) + 1.0 / (RRF_K + rank)

    by_path = {annotation.path: annotation for annotation in corpus}
    by_path.update({path: annotation for path, (annotation, _) in semantic.items()})

    ranked = sorted(fused, key=lambda path: fused[path], reverse=True)[:n]
    return [(by_path[path], semantic[path][1] if path in semantic else math.inf) for path in ranked]
