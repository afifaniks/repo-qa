# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Any, Dict, List, Optional, Sequence

import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from repoqa.storage.vector_store import VectorStore


class LangChainChromaStore(VectorStore):
    """LangChain-compatible ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "repo_qa",
        persist_directory: Optional[str] = "./chroma_data",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize LangChain ChromaDB store.

        Args:
            collection_name: Name of the collection.
            persist_directory: Directory to persist the database.
            embedding_model_name: Name of the embedding model.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name, model_kwargs={"device": device}
        )

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

    def add(
        self,
        embeddings: Sequence[Sequence[float]],
        metadata: Sequence[Dict[str, Any]],
    ) -> None:
        """Add embeddings and metadata to the store.

        Args:
            embeddings: Sequence of embedding vectors.
            metadata: Sequence of metadata dictionaries.
        """
        documents = []
        for meta in metadata:
            content = meta.get("content") or ""
            doc_metadata = {k: v for k, v in meta.items() if k != "content"}
            documents.append(Document(page_content=str(content), metadata=doc_metadata))

        # Add documents (LangChain will handle embeddings)
        self.vectorstore.add_documents(documents)

    def query(self, embedding: Sequence[float], k: int = 5) -> List[Dict[str, Any]]:
        """Query the vector store using embedding.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of matching documents with metadata.
        """
        docs = self.vectorstore.similarity_search_by_vector(embedding, k=k)

        results = []
        for doc in docs:
            result = {"content": doc.page_content, **doc.metadata}
            results.append(result)

        return results

    def get_langchain_retriever(self, k: int = 5):
        """Get a LangChain retriever for RAG chains.

        Args:
            k: Number of documents to retrieve.

        Returns:
            LangChain retriever object.
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
