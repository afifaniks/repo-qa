# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Repository code indexing and embedding module."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from repoqa.embedding.embedding_model import EmbeddingModel


class RepoIndexer(ABC):
    """Indexes and embeds repository code."""

    def __init__(self, embedding_model: EmbeddingModel):
        """Initialize the code indexer.

        Args:
            embedding_model: Instance of the embedding model to use.
        """
        self.embedding_model = embedding_model

    @abstractmethod
    def index_repository(self, repo_path: str) -> Dict[str, Any]:
        """Index a repository and generate embeddings.

        Args:
            repo_path: Path to the repository to index.

        Returns:
            Dictionary containing indexed data and embeddings.
        """
        raise NotImplementedError()
