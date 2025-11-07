# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for code retriever."""

from unittest.mock import Mock

import pytest


class TestCodeRetriever:
    """Test suite for CodeRetriever."""

    def test_initialization(self):
        """Test retriever initialization."""
        from repoqa.retrieval.retriever import CodeRetriever

        mock_vector_store = Mock()
        retriever = CodeRetriever(vector_store=mock_vector_store)

        assert retriever.vector_store == mock_vector_store

    def test_retrieve_basic(self):
        """Test basic retrieval functionality."""
        from repoqa.retrieval.retriever import CodeRetriever

        mock_vector_store = Mock()
        mock_results = [
            {"file": "test1.py", "content": "code1", "score": 0.1},
            {"file": "test2.py", "content": "code2", "score": 0.2},
        ]
        mock_vector_store.search.return_value = mock_results

        retriever = CodeRetriever(vector_store=mock_vector_store)

        query_embedding = [0.1, 0.2, 0.3]
        results = retriever.retrieve(query_embedding, k=2)

        assert results == mock_results
        mock_vector_store.search.assert_called_once_with(query_embedding, top_k=2)

    def test_retrieve_default_k(self):
        """Test retrieval with default k value."""
        from repoqa.retrieval.retriever import CodeRetriever

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []

        retriever = CodeRetriever(vector_store=mock_vector_store)

        query_embedding = [0.1, 0.2, 0.3]
        retriever.retrieve(query_embedding)

        # Default k should be 5
        mock_vector_store.search.assert_called_once_with(query_embedding, top_k=5)

    def test_retrieve_custom_k(self):
        """Test retrieval with custom k value."""
        from repoqa.retrieval.retriever import CodeRetriever

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []

        retriever = CodeRetriever(vector_store=mock_vector_store)

        query_embedding = [0.1, 0.2, 0.3]
        retriever.retrieve(query_embedding, k=10)

        mock_vector_store.search.assert_called_once_with(query_embedding, top_k=10)

    def test_retrieve_empty_results(self):
        """Test retrieval with no results."""
        from repoqa.retrieval.retriever import CodeRetriever

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []

        retriever = CodeRetriever(vector_store=mock_vector_store)

        query_embedding = [0.1, 0.2, 0.3]
        results = retriever.retrieve(query_embedding)

        assert results == []
