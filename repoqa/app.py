# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Main RepoQA application module."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from repoqa.embedding import EmbeddingModel, SentenceTransformerEmbedding
from repoqa.generation.generator import AnswerGenerator
from repoqa.indexing.git_indexer import GitRepoIndexer
from repoqa.indexing.indexer import RepoIndexer
from repoqa.retrieval.retriever import CodeRetriever
from repoqa.storage import ChromaVectorStore, VectorStore


class RepoQA:
    """Main RepoQA application class for code-based question answering."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        repo_indexer: Optional[RepoIndexer] = None,
        llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        vector_store: Optional[VectorStore] = None,
        persist_directory: Optional[str] = "./chroma_data",
        collection_name: str = "repo_qa",
    ):
        """Initialize RepoQA with customizable components.

        Args:
            embedding_model: Model for computing text embeddings.
                If None, uses default SentenceTransformer.
            repo_indexer: Repository indexer component.
                If None, uses GitRepoIndexer.
            llm_model: Name/path of the LLM for answer generation.
            vector_store: Vector store component.
                If None, uses ChromaDB.
            persist_directory: Directory for vector store persistence.
            collection_name: Name for the vector store collection.
        """
        # Initialize embedding model
        self.embedding_model = embedding_model or SentenceTransformerEmbedding()

        # Initialize indexer
        self.indexer = repo_indexer or GitRepoIndexer(self.embedding_model)

        # Initialize vector store
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=collection_name, persist_directory=persist_directory
        )

        # Initialize retriever and generator
        self.retriever = CodeRetriever(self.vector_store)
        self.generator = AnswerGenerator(llm_model)

    def index_repository(
        self,
        repo_path: Union[str, Path],
        clone_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a repository and store embeddings.

        Args:
            repo_path: Path/URL of the repository to index.
            clone_dir: Optional directory to clone into.

        Returns:
            Dictionary with indexing results and metadata.
        """
        # Index repository
        result = self.indexer.index_repository(
            repo_path=str(repo_path),
            clone_dir=clone_dir,
        )

        # Store embeddings
        self.vector_store.add(
            embeddings=result["embeddings"],
            metadata=[
                {
                    "file_path": chunk.file_path,
                    "content": chunk.content,
                }
                for chunk in result["chunks"]
            ],
        )

        return result

    def ask(self, query: str, max_results: int = 5) -> str:
        """Answer a question about the repository.

        Args:
            query: Natural language query about the repository.
            max_results: Maximum number of code snippets to retrieve.

        Returns:
            Generated answer based on repository context.
        """
        # Get relevant code context using query embedding
        # Get single vector embedding
        query_embedding = self.embedding_model.encode(query)[0]
        context = self.retriever.retrieve(query_embedding, k=max_results)

        # Generate answer using retrieved context
        return self.generator.generate_answer(query, {"matches": context})


if __name__ == "__main__":
    # Initialize RepoQA with default settings
    repo_qa = RepoQA(
        persist_directory="./chroma_data",
        collection_name="demo_repo",
        llm_model="unsloth/gemma-3-1b-it",
    )

    # Index a repository
    repo_qa.index_repository(
        repo_path="git@github.com:afifaniks/ubuntu_autolocker.git",
        clone_dir="./repo_data",
    )

    # Ask questions
    questions = [
        "How is face lock implemented?",
        "What languages is the project written in?",
        "Are there any security features?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        print(f"A: {repo_qa.ask(question)}")
