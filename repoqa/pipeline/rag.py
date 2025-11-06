# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from repoqa.embedding import SentenceTransformerEmbedding
from repoqa.indexing.git_indexer import GitRepoIndexer
from repoqa.pipeline.pipeline import Pipeline
from repoqa.pipeline.prompts import BASIC_RAG_PROMPT


class RAGPipeline(Pipeline):
    """Complete RAG pipeline using LangChain components."""

    def __init__(
        self,
        llm_model: Any,
        embedding_model: str = "all-mpnet-base-v2",
        persist_directory: str = "./chroma_data",
        collection_name: str = "repo_qa",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ):
        """Initialize the RAG pipeline.

        Args:
            llm_model: Name of the Ollama model (e.g., 'llama3.2:3b', 'codellama:7b').
            embedding_model: Name of the embedding model.
            persist_directory: Directory to persist vector store.
            collection_name: Name of the vector store collection.
            ollama_base_url: Base URL for Ollama server.
            temperature: Sampling temperature.
        """
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.llm = llm_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        # Create prompt template
        self.prompt = PromptTemplate.from_template(BASIC_RAG_PROMPT)

        self.safe_retriever = self._safe_retriever

        self.rag_chain = (
            {"context": self._retrieve_and_format, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Initialize indexer for repository processing
        self.embedding_model_obj = SentenceTransformerEmbedding(
            model_name=embedding_model
        )
        self.indexer = GitRepoIndexer(self.embedding_model_obj)

        # Track source files for attribution
        self.source_files = []

    def _safe_retriever(self, query):
        """Safe retrieval that filters out invalid documents."""
        try:
            # Clear previous source files
            self.source_files = []

            # Use LangChain's built-in similarity search
            docs = self.vectorstore.similarity_search(query, k=5)

            # Filter out invalid documents and track sources
            valid_docs = []
            for i, doc in enumerate(docs):
                if (
                    doc.page_content is None
                    or not isinstance(doc.page_content, str)
                    or not doc.page_content.strip()
                ):
                    logger.debug(f"Skipping invalid document {i}")
                    continue

                # Track source file
                file_path = doc.metadata.get("file_path", "unknown")
                if file_path != "unknown" and file_path not in self.source_files:
                    self.source_files.append(file_path)

                valid_docs.append(doc)

            logger.info(f"Retrieved {len(valid_docs)} valid documents")
            logger.info(f"Source files: {self.source_files}")
            return valid_docs

        except Exception as e:
            logger.error(f"Safe retrieval failed: {e}")
            return []

    def _format_docs(self, docs):
        """Format retrieved documents."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            # Safety check for document content
            if doc.page_content is None:
                logger.debug(f"Skipping document {i} with None content")
                continue

            if not isinstance(doc.page_content, str):
                logger.debug(f"Skipping document {i} with invalid type")
                continue

            if not doc.page_content.strip():
                logger.debug(f"Skipping document {i} with empty content")
                continue

            file_path = doc.metadata.get("file_path", "unknown")
            content = doc.page_content.strip()
            formatted.append(
                (
                    f"File {i}: {file_path}\n"  # header
                    f"```\n{content}\n```\n"  # fenced block
                )
            )

        if not formatted:
            return "No valid documents found for context."

        return "\n".join(formatted)

    def _retrieve_and_format(self, query):
        docs = self._safe_retriever(query)
        return self._format_docs(docs)

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
            Indexing results.
        """
        # Index repository using existing indexer
        result = self.indexer.index_repository(
            repo_path=str(repo_path),
            clone_dir=clone_dir,
        )

        # Convert chunks to LangChain documents
        documents = []
        logger.info(f"🔍 Processing {len(result.get('chunks', []))} chunks...")

        for i, chunk in enumerate(result.get("chunks", [])):
            # Validate chunk structure
            if not hasattr(chunk, "content") or not hasattr(chunk, "file_path"):
                logger.warning(f"Chunk {i} missing required attributes")
                continue

            # Validate content is not None and is a string
            content = chunk.content
            if content is None:
                logger.warning(f"Chunk {i} has None content from {chunk.file_path}")
                continue

            if not isinstance(content, str):
                logger.warning(f"Chunk {i} has non-string content: {type(content)}")
                continue

            # Ensure content is not empty
            if not content.strip():
                logger.warning(f"Skipping empty chunk: {chunk.file_path}")
                continue

            try:
                doc = Document(
                    page_content=content.strip(),
                    metadata={
                        "file_path": chunk.file_path or "unknown",
                    },
                )
                documents.append(doc)
            except (ValueError, TypeError) as e:
                logger.error(f"Error creating document from chunk {i}: {e}")
                logger.error(f"   Content type: {type(content)}")
                logger.error(f"   Content preview: {repr(content[:100])}")
                continue

        # Add documents to vector store
        if documents:
            self.vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")

        return {
            "status": "success",
            "documents_added": len(documents),
            "chunks_processed": len(result["chunks"]),
        }

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response.

        Args:
            response: Raw response from the LLM.
        Returns:
            Cleaned response string.
        """
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return response.strip()

    def ask(self, query: str) -> str:
        """Ask a question about the indexed repository.

        Args:
            query: User's question.

        Returns:
            Generated answer based on repository context.
        """
        try:
            logger.info(f"Processing query: {query}")

            # Use the safe retriever to avoid invalid documents from
            # older indexes
            docs = self.safe_retriever(query)
            logger.info(f"Retrieved {len(docs)} documents")

            for i, doc in enumerate(docs):
                if doc.page_content is None:
                    logger.warning(f"Document {i} has None page_content!")

            response = self.rag_chain.invoke(query)
            if response:
                answer = self._clean_response(response)

                # Append source files if any were retrieved
                if self.source_files:
                    answer += "\n\n---\n**Sources:**\n" + "\n".join(
                        f"- {src}" for src in self.source_files
                    )

                return answer
            return "I couldn't generate a response."
        except Exception as e:
            return f"Error generating response: {str(e)}"
