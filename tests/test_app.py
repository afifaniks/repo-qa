# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for main RepoQA application."""

from unittest.mock import Mock, patch

import pytest


class TestRepoQA:
    """Test suite for RepoQA main application class."""

    @patch("repoqa.app.RAGPipeline")
    @patch("repoqa.app.SentenceTransformerEmbedding")
    @patch("repoqa.app.GitRepoIndexer")
    def test_initialization_rag_mode(
        self, mock_indexer_class, mock_embedding_class, mock_pipeline_class
    ):
        """Test RepoQA initialization in RAG mode."""
        from repoqa.app import RepoQA

        mock_llm = Mock()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        repo_qa = RepoQA(
            llm_model=mock_llm,
            embedding_model="test-model",
            collection_name="test-collection",
            collection_chunk_size=1024,
            ollama_base_url="http://localhost:11434",
            mode="rag",
            repo_path="./test_repo",
            persist_directory="./chroma_data",
            temperature=0.5,
        )

        assert repo_qa.mode == "rag"
        assert repo_qa.pipeline == mock_pipeline
        mock_pipeline_class.assert_called_once()

    @patch("repoqa.app.AgenticRAGPipeline")
    @patch("repoqa.app.SentenceTransformerEmbedding")
    @patch("repoqa.app.GitRepoIndexer")
    def test_initialization_agent_mode(
        self, mock_indexer_class, mock_embedding_class, mock_pipeline_class
    ):
        """Test RepoQA initialization in agent mode."""
        from repoqa.app import RepoQA

        mock_llm = Mock()
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline

        repo_qa = RepoQA(
            llm_model=mock_llm,
            embedding_model="test-model",
            collection_name="test-collection",
            collection_chunk_size=1024,
            ollama_base_url="http://localhost:11434",
            mode="agent",
            repo_path="./test_repo",
            persist_directory="./chroma_data",
            temperature=0.5,
        )

        assert repo_qa.mode == "agent"
        assert repo_qa.pipeline == mock_pipeline
        mock_pipeline_class.assert_called_once()

    @patch("repoqa.app.RAGPipeline")
    @patch("repoqa.app.SentenceTransformerEmbedding")
    @patch("repoqa.app.GitRepoIndexer")
    def test_initialization_invalid_mode(
        self, mock_indexer_class, mock_embedding_class, mock_pipeline_class
    ):
        """Test RepoQA initialization with invalid mode."""
        from repoqa.app import RepoQA

        mock_llm = Mock()

        with pytest.raises(ValueError) as exc_info:
            RepoQA(
                llm_model=mock_llm,
                embedding_model="test-model",
                collection_name="test-collection",
                collection_chunk_size=1024,
                ollama_base_url="http://localhost:11434",
                mode="invalid_mode",
                repo_path="./test_repo",
                persist_directory="./chroma_data",
                temperature=0.5,
            )

        assert "Unsupported mode" in str(exc_info.value)

    @patch("repoqa.app.RAGPipeline")
    @patch("repoqa.app.SentenceTransformerEmbedding")
    @patch("repoqa.app.GitRepoIndexer")
    def test_index_repository(
        self, mock_indexer_class, mock_embedding_class, mock_pipeline_class
    ):
        """Test repository indexing."""
        from repoqa.app import RepoQA

        mock_llm = Mock()
        mock_pipeline = Mock()
        mock_pipeline.index_repository.return_value = {
            "status": "success",
            "documents_added": 100,
            "chunks_processed": 100,
        }
        mock_pipeline_class.return_value = mock_pipeline

        repo_qa = RepoQA(
            llm_model=mock_llm,
            embedding_model="test-model",
            collection_name="test-collection",
            collection_chunk_size=1024,
            ollama_base_url="http://localhost:11434",
            mode="rag",
            repo_path="./test_repo",
            persist_directory="./chroma_data",
            temperature=0.5,
        )

        result = repo_qa.index_repository(
            repo_path="https://github.com/test/repo.git",
            clone_dir="./repo_data",
        )

        assert result["status"] == "success"
        assert result["documents_added"] == 100
        mock_pipeline.index_repository.assert_called_once_with(
            "https://github.com/test/repo.git", "./repo_data"
        )

    @patch("repoqa.app.RAGPipeline")
    @patch("repoqa.app.SentenceTransformerEmbedding")
    @patch("repoqa.app.GitRepoIndexer")
    def test_ask(self, mock_indexer_class, mock_embedding_class, mock_pipeline_class):
        """Test asking a question."""
        from repoqa.app import RepoQA

        mock_llm = Mock()
        mock_pipeline = Mock()
        mock_pipeline.ask.return_value = "This is the answer."
        mock_pipeline_class.return_value = mock_pipeline

        repo_qa = RepoQA(
            llm_model=mock_llm,
            embedding_model="test-model",
            collection_name="test-collection",
            collection_chunk_size=1024,
            ollama_base_url="http://localhost:11434",
            mode="rag",
            repo_path="./test_repo",
            persist_directory="./chroma_data",
            temperature=0.5,
        )

        answer = repo_qa.ask("What is this repository about?")

        assert answer == "This is the answer."
        mock_pipeline.ask.assert_called_once_with("What is this repository about?")
