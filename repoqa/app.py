# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from repoqa.llm.llm_factory import get_llm
from repoqa.util.setup_util import setup

setup()

from repoqa.pipeline.langchain_rag import LangChainRAGPipeline  # noqa: E402


class RepoQA:
    """Main RepoQA application class for code-based question answering."""

    def __init__(
        self,
        llm_model: str = "gemma3:1b",
        persist_directory: Optional[str] = "./chroma_data",
        collection_name: str = "repo_qa",
        ollama_base_url: str = "http://localhost:11435",
    ):
        """Initialize RepoQA with customizable components.

        Args:
            llm_model: Name of the Ollama model (e.g., 'llama3.2:3b', 'codellama:7b').
            persist_directory: Directory for vector store persistence.
            collection_name: Name for the vector store collection.
            ollama_base_url: Base URL for Ollama server.
        """

        logger.info("Initializing RAG pipeline...")
        self.rag_pipeline = LangChainRAGPipeline(
            llm_model=llm_model,
            persist_directory=persist_directory or "./chroma_data",
            collection_name=collection_name,
            ollama_base_url=ollama_base_url,
            temperature=0.3,  # More focused responses
        )

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
        return self.rag_pipeline.index_repository(repo_path, clone_dir)

    def ask(self, query: str) -> str:
        """Answer a question about the repository.

        Args:
            query: Natural language query about the repository.

        Returns:
            Generated answer based on repository context.
        """
        return self.rag_pipeline.ask(query)


if __name__ == "__main__":
    repo_qa = RepoQA(
        persist_directory="./chroma_data",
        collection_name="demo_repo",
        llm_model=get_llm(model_name="gemma3:1b", backend="ollama"),
    )

    # Index a repository
    print("Indexing repository...")
    result = repo_qa.index_repository(
        repo_path="https://github.com/afifaniks/repo-qa",
        clone_dir="./repo_data",
    )
    print(f"Indexing completed: {result}")

    # Ask questions
    questions = [
        "What is this project about?",
        "How do I run this app?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = repo_qa.ask(question)
        print(f"A: {answer}")
