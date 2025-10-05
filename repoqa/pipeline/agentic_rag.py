# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from repoqa.embedding import SentenceTransformerEmbedding
from repoqa.indexing.git_indexer import GitRepoIndexer
from repoqa.pipeline.pipeline import Pipeline


class AgenticRAGPipeline(Pipeline):
    """Hybrid agent combining file exploration with RAG semantic search."""

    def __init__(
        self,
        llm_model: Any,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_data",
        collection_name: str = "repo_qa",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        repo_path: str = "./repo_data",
    ):
        """Initialize the hybrid RAG-Agent pipeline.

        Args:
            llm_model: A supported llm model.
            embedding_model: Name of the embedding model.
            persist_directory: Directory to persist vector store.
            collection_name: Name of the vector store collection.
            ollama_base_url: Base URL for Ollama server.
            temperature: Sampling temperature.
            repo_path: Path to the repository to explore.
        """
        self.llm = llm_model
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.repo_path = Path(repo_path)

        # Initialize embeddings and vector store for RAG
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        self.embedding_model_obj = SentenceTransformerEmbedding(
            model_name=embedding_model
        )
        self.indexer = GitRepoIndexer(self.embedding_model_obj)
        self.tools = self._create_tools()

        # Create the agent
        self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """Create tools for both RAG and file exploration."""

        # RAG-based tools
        def semantic_search(query: str, k: int = 5) -> str:
            """Search for relevant code using semantic similarity."""
            try:
                docs = self.vectorstore.similarity_search(query, k=k)
                if not docs:
                    return f"No relevant documents found for: {query}"

                results = []
                for i, doc in enumerate(docs, 1):
                    file_path = doc.metadata.get("file_path", "unknown")
                    content = doc.page_content.strip()
                    results.append(f"Result {i} from {file_path}:\n```\n{content}\n```")

                return f"Semantic search results for '{query}':\n\n" + "\n\n".join(
                    results
                )

            except Exception as e:
                return f"Semantic search failed: {e}"

        def similarity_search_with_score(query: str, k: int = 3) -> str:
            """Search with similarity scores to show relevance."""
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(
                    query, k=k
                )
                if not docs_with_scores:
                    return f"No relevant documents found for: {query}"

                results = []
                for i, (doc, score) in enumerate(docs_with_scores, 1):
                    file_path = doc.metadata.get("file_path", "unknown")
                    content = doc.page_content.strip()
                    results.append(
                        f"Result {i} (score: {score:.3f}) from {file_path}:\n```\n{content}\n```"
                    )

                return f"Scored search results for '{query}':\n\n" + "\n\n".join(
                    results
                )

            except Exception as e:
                return f"Scored search failed: {e}"

        def list_directory(path: str = "") -> str:
            """List files and directories in the given path."""
            try:
                target_path = self.repo_path / path if path else self.repo_path
                if not target_path.exists():
                    return f"Path does not exist: {target_path}"

                items = []
                for item in sorted(target_path.iterdir()):
                    if item.name.startswith("."):
                        continue

                    item_type = "[DIR]" if item.is_dir() else "[FILE]"
                    rel_path = item.relative_to(self.repo_path)
                    items.append(f"{item_type} {rel_path}")

                if not items:
                    return f"Empty directory: {target_path}"

                return f"Directory listing for {target_path}:\n" + "\n".join(items)

            except Exception as e:
                return f"Error listing directory: {e}"

        def read_file(file_path: str) -> str:
            """Read the content of a file."""
            try:
                target_file = self.repo_path / file_path
                if not target_file.exists():
                    return f"File does not exist: {target_file}"

                if not target_file.is_file():
                    return f"Path is not a file: {target_file}"

                if target_file.stat().st_size > 100000:
                    return f"File too large: {target_file}"

                with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                return f"Content of {file_path}:\n```\n{content}\n```"

            except Exception as e:
                return f"Error reading file: {e}"

        def find_files(pattern: str = "", extension: str = "") -> str:
            """Find files by pattern or extension."""
            try:
                matches = []
                for item in self.repo_path.rglob("*"):
                    if item.is_file() and not item.name.startswith("."):
                        rel_path = item.relative_to(self.repo_path)

                        if extension and not item.name.endswith(f".{extension}"):
                            continue

                        if pattern and pattern.lower() not in item.name.lower():
                            continue

                        matches.append(str(rel_path))

                if not matches:
                    return "No files found matching criteria"

                matches.sort()
                return f"Found {len(matches)} files:\n" + "\n".join(
                    f"📄 {f}" for f in matches[:20]
                )

            except Exception as e:
                return f"Error finding files: {e}"

        return [
            Tool(
                name="semantic_search",
                description="Search for relevant code semantically",
                func=semantic_search,
            ),
            Tool(
                name="similarity_search_with_score",
                description="Search with similarity scores",
                func=similarity_search_with_score,
            ),
            Tool(
                name="list_directory",
                description="List files and directories",
                func=list_directory,
            ),
            Tool(
                name="read_file",
                description="Read file content",
                func=read_file,
            ),
            Tool(
                name="find_files",
                description="Find files by pattern or extension",
                func=find_files,
            ),
        ]

    def _create_agent(self):
        prompt = hub.pull("hwchase17/react")
        self.agent = create_react_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

    def index_repository(
        self,
        repo_path: Union[str, Path],
        clone_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index repository for RAG and set path for file exploration."""
        # First, run the indexer to get the repository
        result = self.indexer.index_repository(
            repo_path=str(repo_path),
            clone_dir=clone_dir,
        )

        # Set the correct repository path for file exploration
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

        documents = []
        chunks = result.get("chunks", [])

        for chunk in chunks:
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

        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

        if not self.repo_path.exists():
            logger.error(f"Repository path does not exist: {self.repo_path}")
        else:
            logger.info(f"Repository indexed at: {self.repo_path}")

        return {
            "status": "success",
            "documents_added": len(documents),
            "chunks_processed": len(result["chunks"]),
            "repo_path": str(self.repo_path),
            "rag_enabled": True,
            "file_exploration_enabled": True,
        }

    def ask(self, query: str) -> str:
        """Ask the hybrid RAG-Agent a question."""
        try:
            logger.info(f"Agent processing: {query}")

            if not self.repo_path.exists():
                return f"Repository path does not exist: {self.repo_path}"

            response = self.agent_executor.invoke({"input": query})
            return response.get("output", "No response generated")

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return f"Error: {str(e)}"
