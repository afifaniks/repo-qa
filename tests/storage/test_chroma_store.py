# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for ChromaDB vector store."""

import sys
from unittest.mock import MagicMock, Mock, patch

# Mock chromadb module completely before any imports
chromadb_mock = MagicMock()
chromadb_mock.config = MagicMock()
chromadb_mock.config.Settings = MagicMock(return_value={})
chromadb_mock.Client = MagicMock()
chromadb_mock.PersistentClient = MagicMock()
chromadb_mock.api = MagicMock()
chromadb_mock.api.CreateCollectionConfiguration = MagicMock()
sys.modules["chromadb"] = chromadb_mock
sys.modules["chromadb.config"] = chromadb_mock.config
sys.modules["chromadb.api"] = chromadb_mock.api

import numpy as np
import pytest

from repoqa.storage.chroma_store import ChromaVectorStore


class TestChromaVectorStore:
    """Test suite for ChromaVectorStore."""

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_initialization_memory(self, mock_client_class):
        """Test initialization with in-memory client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = ChromaVectorStore(collection_name="test-collection")

        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            name="test-collection"
        )

    @patch("repoqa.storage.chroma_store.chromadb.PersistentClient")
    def test_initialization_persistent(self, mock_persistent_client_class):
        """Test initialization with persistent client."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_persistent_client_class.return_value = mock_client

        store = ChromaVectorStore(
            collection_name="test-collection",
            persist_directory="/path/to/db",
        )

        assert store.client == mock_client
        assert store.collection == mock_collection
        mock_persistent_client_class.assert_called_once_with(path="/path/to/db")

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_add_embeddings(self, mock_client_class):
        """Test adding embeddings and metadata."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = ChromaVectorStore()

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [
            {"file": "test1.py", "content": "code1"},
            {"file": "test2.py", "content": "code2"},
        ]

        store.add(embeddings, metadata)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["embeddings"] == embeddings
        assert call_kwargs["metadatas"] == metadata
        assert len(call_kwargs["ids"]) == 2

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_add_embeddings_length_mismatch(self, mock_client_class):
        """Test that adding mismatched embeddings and metadata raises error."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        store = ChromaVectorStore()

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"file": "test1.py"}]  # Only one metadata

        with pytest.raises(ValueError) as exc_info:
            store.add(embeddings, metadata)

        assert "same length" in str(exc_info.value)

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_search_basic(self, mock_client_class):
        """Test basic search functionality."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # Mock search results
        mock_collection.query.return_value = {
            "metadatas": [
                [
                    {"file": "test1.py", "content": "code1"},
                    {"file": "test2.py", "content": "code2"},
                ]
            ],
            "distances": [[0.1, 0.2]],
        }

        store = ChromaVectorStore()

        query_embedding = [0.1, 0.2, 0.3]
        results = store.search(query_embedding, top_k=2)

        assert len(results) == 2
        assert results[0]["file"] == "test1.py"
        assert results[0]["score"] == 0.1
        assert results[0]["content"] == "code1"
        assert results[1]["file"] == "test2.py"
        assert results[1]["score"] == 0.2

        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=2,
        )

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_search_with_filter(self, mock_client_class):
        """Test search with metadata filter."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        mock_collection.query.return_value = {
            "metadatas": [[{"file": "test1.py", "content": "code1"}]],
            "distances": [[0.1]],
        }

        store = ChromaVectorStore()

        query_embedding = [0.1, 0.2, 0.3]
        metadata_filter = {"file": "test1.py"}
        results = store.search(
            query_embedding, top_k=5, metadata_filter=metadata_filter
        )

        assert len(results) == 1
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=5,
            where=metadata_filter,
        )

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_search_flatten_2d_embedding(self, mock_client_class):
        """Test that 2D embeddings are flattened."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        mock_collection.query.return_value = {
            "metadatas": [[{"file": "test.py", "content": "code"}]],
            "distances": [[0.1]],
        }

        store = ChromaVectorStore()

        # Test with numpy 2D array
        query_embedding = np.array([[0.1, 0.2, 0.3]])
        store.search(query_embedding)

        # Should flatten to 1D
        call_args = mock_collection.query.call_args[1]
        assert len(call_args["query_embeddings"]) == 1
        assert isinstance(call_args["query_embeddings"][0], (list, np.ndarray))

    @patch("repoqa.storage.chroma_store.chromadb.Client")
    def test_search_flatten_nested_list(self, mock_client_class):
        """Test that nested list embeddings are flattened."""

        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        mock_collection.query.return_value = {
            "metadatas": [[{"file": "test.py", "content": "code"}]],
            "distances": [[0.1]],
        }

        store = ChromaVectorStore()

        # Test with nested list
        query_embedding = [[0.1, 0.2, 0.3]]
        store.search(query_embedding)

        # Should flatten to 1D
        call_args = mock_collection.query.call_args[1]
        assert call_args["query_embeddings"] == [[0.1, 0.2, 0.3]]
