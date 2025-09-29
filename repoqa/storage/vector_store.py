# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class VectorStore(ABC):
    """Abstract base class for vector stores that persist embeddings."""

    @abstractmethod
    def add(
        self, embeddings: Sequence[Sequence[float]], metadata: Sequence[Dict[str, Any]]
    ) -> None:
        """Add embeddings and metadata to the store.

        Args:
            embeddings: List of embedding vectors to store
            metadata: List of metadata dictionaries for each embedding
        """

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding.

        Args:
            query_embedding: Embedding vector to search with
            top_k: Number of results to return
            metadata_filter: Optional metadata filter conditions

        Returns:
            List of dictionaries containing similar documents with metadata
        """
