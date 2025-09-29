# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Any, Dict, List

from repoqa.storage.vector_store import VectorStore


class CodeRetriever:
    """Retrieves relevant code snippets based on queries."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the code retriever.

        Args:
            vector_store: Vector store containing indexed repository data.
        """
        self.vector_store = vector_store

    def retrieve(
        self, query_embedding: List[float], k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant code snippets for a query.

        Args:
            query_embedding: Query embedding vector.
            k: Number of snippets to retrieve.

        Returns:
            List of relevant code snippets with metadata.
        """
        return self.vector_store.search(query_embedding, top_k=k)
