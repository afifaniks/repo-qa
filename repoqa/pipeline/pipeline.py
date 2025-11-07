# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain_core.documents import Document
from loguru import logger


class Pipeline(ABC):
    """Base class for RAG pipelines with shared indexing logic."""

    indexer: Any
    vectorstore: Any
    repo_path: Optional[Path]

    def index_repository(
        self,
        repo_path: Union[str, Path],
        clone_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a repository and add to vector store.

        Args:
            repo_path: Path or URL of the repository.
            clone_dir: Directory to clone into.

        Returns:
            Indexing results with status and statistics.
        """
        # Index repository using the indexer
        result = self.indexer.index_repository(
            repo_path=str(repo_path),
            clone_dir=clone_dir,
        )

        # Set the correct repository path for file exploration (for agentic)
        if hasattr(self, "repo_path"):
            if clone_dir:
                base_path = Path(clone_dir)
                if str(repo_path).startswith(("http", "git")):
                    subdirs = [
                        d
                        for d in base_path.iterdir()
                        if d.is_dir() and not d.name.startswith(".")
                    ]
                    if subdirs:
                        self.repo_path = subdirs[0].resolve()
                    else:
                        self.repo_path = base_path.resolve()
                else:
                    self.repo_path = base_path.resolve()
            else:
                self.repo_path = Path(repo_path).resolve()

            if not self.repo_path.exists():
                logger.error(f"Repository path does not exist: {self.repo_path}")
            else:
                logger.info(f"Repository indexed at: {self.repo_path}")

        # Convert chunks to LangChain documents
        documents = []
        chunks = result.get("chunks", [])
        logger.info(f"Processing {len(chunks)} chunks...")

        for chunk in chunks:
            # Validate chunk structure
            if not hasattr(chunk, "content") or not hasattr(chunk, "file_path"):
                continue

            content = chunk.content
            if not content or not isinstance(content, str) or not content.strip():
                continue

            try:
                doc = Document(
                    page_content=content.strip(),
                    metadata={"file_path": chunk.file_path or "unknown"},
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Error creating document: {e}")
                continue

        # Add documents to vector store
        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

        return {
            "status": "success",
            "documents_added": len(documents),
            "chunks_processed": len(chunks),
            "repo_path": str(getattr(self, "repo_path", repo_path)),
            "rag_enabled": True,
            "file_exploration_enabled": hasattr(self, "repo_path"),
        }

    @abstractmethod
    def ask(self, query: str) -> str:
        """Ask a question about the indexed repository."""
        raise NotImplementedError()
