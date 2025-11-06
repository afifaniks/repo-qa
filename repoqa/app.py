# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from repoqa.llm.llm_factory import get_llm
from repoqa.util.setup_util import setup

setup()

from repoqa.pipeline.agentic_rag import AgenticRAGPipeline  # noqa: E402
from repoqa.pipeline.rag import RAGPipeline  # noqa: E402


class RepoQA:
    """Main RepoQA application class for code-based question answering."""

    def __init__(
        self,
        llm_model: Any,
        persist_directory: Optional[str] = "./chroma_data",
        collection_name: str = "repo_qa",
        ollama_base_url: str = "http://localhost:11435",
        mode: str = "hybrid",  # "rag", "agent", or "hybrid"
    ):
        """Initialize RepoQA with customizable components.

        Args:
            llm_model: A supported llm model.
            persist_directory: Directory for vector store persistence.
            collection_name: Name for the vector store collection.
            ollama_base_url: Base URL for Ollama server.
            mode: "rag" (vector search), "agent" (file tools), or
                "hybrid" (RAG + tools).
        """
        self.mode = mode

        if mode == "agent":
            logger.info("Initializing Agent pipeline...")
            self.pipeline = AgenticRAGPipeline(
                llm_model=llm_model,
                persist_directory=persist_directory or "./chroma_data",
                collection_name=collection_name,
                ollama_base_url=ollama_base_url,
                temperature=0.3,
            )
        else:
            logger.info("Initializing RAG pipeline...")
            self.pipeline = RAGPipeline(
                llm_model=llm_model,
                persist_directory=persist_directory or "./chroma_data",
                collection_name=collection_name,
                ollama_base_url=ollama_base_url,
                temperature=0.3,
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
        return self.pipeline.index_repository(repo_path, clone_dir)

    def ask(self, query: str) -> str:
        """Answer a question about the repository.

        Args:
            query: Natural language query about the repository.

        Returns:
            Generated answer based on repository context.
        """
        return self.pipeline.ask(query)


if __name__ == "__main__":
    # Use agent mode by default - much more powerful for code exploration
    repo_qa = RepoQA(
        persist_directory="./chroma_data",
        collection_name="demo_repo",
        llm_model=get_llm("qwen3:1.7b", backend="ollama"),
        mode="agent",
    )

    # Index a repository
    print("Indexing repository...")
    result = repo_qa.index_repository(
        repo_path="git@github.com:afifaniks/repo-qa.git",
        clone_dir="./repo_data",
    )
    print(f"Indexing completed: {result}")

    # Ask questions
    questions = [
        # "What is this project about?",
        "What kind of licenses are permitted for dependencies?",
        # "How to run this application?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = repo_qa.ask(question)
        print(f"A: {answer}")
