import uuid
from pathlib import Path

import chromadb
import chromadb.api.models.Collection
import chromadb.api.types
from chromadb.utils import embedding_functions

from ..impl import Annotation

TextOrEmbedding = str | list[float]


class Database:
    def __init__(self, path: Path):
        self._client = chromadb.PersistentClient(path=str(path))
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def collection(self, name: str) -> chromadb.api.models.Collection.Collection:
        """Get or create a ChromaDB collection."""
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_function,
        )

    def store(self, collection: str, annotation: Annotation):
        """Store the text embedding in a ChromaDB collection, keyed by annotation path."""
        unique_id = annotation.path or uuid.uuid4().hex

        metadata = dict(annotation.as_metadata)

        self.collection(collection).upsert(documents=[annotation.as_text], metadatas=[metadata], ids=[unique_id])

    def list_all(self, collection: str, where: dict | None = None) -> chromadb.api.types.GetResult:
        """List all paths in the collection."""
        return self.collection(collection).get(where=where)

    def query(self, collection: str, embedding: TextOrEmbedding, num_results: int = 1, where: dict | None = None) -> chromadb.api.types.QueryResult:
        """Query the collection for similar embeddings."""

        if isinstance(embedding, str):
            return self.collection(collection).query(query_texts=[embedding], n_results=num_results, where=where)

        return self.collection(collection).query(query_embeddings=[embedding], n_results=num_results, where=where)

    def _to_embedding(self, value: TextOrEmbedding) -> list[float]:
        if isinstance(value, str):
            return self._embedding_function([value])[0]
        return value

    def get_distance(self, text1: TextOrEmbedding, text2: TextOrEmbedding) -> float:
        """Get distance between two vectors"""
        embedding_text_1 = self._to_embedding(text1)
        embedding_text_2 = self._to_embedding(text2)
        distance = 0
        for i, val1 in enumerate(embedding_text_1):
            val2 = embedding_text_2[i]
            distance += (val1 - val2) ** 2
        return round(distance / 100, 3)
