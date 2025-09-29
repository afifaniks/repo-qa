import uuid
from typing import Any, Dict, List, Optional, Sequence

import chromadb

from repoqa.storage.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Lightweight ChromaDB vector store wrapper."""

    def __init__(
        self,
        collection_name: str = "repo_index",
        persist_directory: Optional[str] = None,
    ):
        self.client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.Client()
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(
        self, embeddings: Sequence[Sequence[float]], metadata: Sequence[Dict[str, Any]]
    ) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")

        ids = [str(uuid.uuid4()) for _ in embeddings]
        self.collection.add(embeddings=embeddings, metadatas=metadata, ids=ids)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings and return metadata + chunk content."""

        # Flatten embedding if [[...]]
        if hasattr(query_embedding, "ndim") and query_embedding.ndim == 2:
            query_embedding = query_embedding[0]
        elif (
            isinstance(query_embedding, list)
            and len(query_embedding) == 1
            and isinstance(query_embedding[0], list)
        ):
            query_embedding = query_embedding[0]

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if metadata_filter:
            query_args["where"] = metadata_filter

        results = self.collection.query(**query_args)

        return [
            {**md, "score": dist, "content": md.get("content", "")}
            for md, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
