# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for RAG pipeline."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies():
    """Auto-mock dependencies for all tests in this module."""
    with patch("repoqa.pipeline.rag.SentenceTransformerEmbedding"), patch(
        "repoqa.pipeline.rag.HuggingFaceEmbeddings"
    ), patch("repoqa.pipeline.rag.Chroma"):
        yield


class TestRAGPipeline:
    """Test suite for RAGPipeline."""

    @patch("repoqa.pipeline.rag.Chroma")
    def test_initialization(
        self,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test RAG pipeline initialization."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        assert pipeline.embedding_model_name == "test-model"
        assert pipeline.collection_name == "test-collection"
        assert pipeline.temperature == 0.5

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_safe_retriever(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test safe document retrieval."""
        from repoqa.pipeline.rag import RAGPipeline

        # Create mock documents
        valid_doc = Document(
            page_content="Valid content",
            metadata={"file_path": "test.py"},
        )
        # Create invalid docs as Mock objects to bypass pydantic validation
        invalid_doc1 = Mock()
        invalid_doc1.page_content = None
        invalid_doc1.metadata = {"file_path": "invalid1.py"}

        invalid_doc2 = Document(
            page_content="",
            metadata={"file_path": "invalid2.py"},
        )

        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [
            valid_doc,
            invalid_doc1,
            invalid_doc2,
        ]
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        docs = pipeline._safe_retriever("test query")

        # Should only return valid document
        assert len(docs) == 1
        assert docs[0].page_content == "Valid content"
        assert "test.py" in pipeline.source_files

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_format_docs(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test document formatting."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        docs = [
            Document(
                page_content="def hello():\n    pass",
                metadata={"file_path": "test1.py"},
            ),
            Document(
                page_content="class MyClass:\n    pass",
                metadata={"file_path": "test2.py"},
            ),
        ]

        formatted = pipeline._format_docs(docs)

        assert "File 1: test1.py" in formatted
        assert "File 2: test2.py" in formatted
        assert "def hello()" in formatted
        assert "class MyClass" in formatted
        assert "```" in formatted  # Check for code fencing

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_format_docs_empty(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test formatting with no valid documents."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        # Use Mock objects to bypass pydantic validation
        doc1 = Mock()
        doc1.page_content = None
        doc1.metadata = {"file_path": "test.py"}

        doc2 = Document(page_content="", metadata={"file_path": "test2.py"})
        docs = [doc1, doc2]

        formatted = pipeline._format_docs(docs)

        assert "No valid documents found" in formatted

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_clean_response(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test response cleaning."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        # Test removing <think> tags
        response = "<think>Internal reasoning</think>This is the answer."
        cleaned = pipeline._clean_response(response)
        assert cleaned == "This is the answer."
        assert "<think>" not in cleaned

        # Test with no tags
        response = "Simple answer"
        cleaned = pipeline._clean_response(response)
        assert cleaned == "Simple answer"

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_ask(
        self,
        mock_embeddings,
        mock_chroma,
        tmp_path,
    ):
        """Test asking a question."""
        from langchain_core.language_models import FakeListLLM

        from repoqa.pipeline.rag import RAGPipeline

        # Mock vectorstore with valid documents
        valid_doc = Document(
            page_content="def add(a, b):\n    return a + b",
            metadata={"file_path": "utils.py"},
        )
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [valid_doc]
        mock_chroma.return_value = mock_vectorstore

        # Use FakeListLLM for proper LCEL chain compatibility
        fake_llm = FakeListLLM(responses=["The add function adds two numbers."])

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=fake_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("What does the add function do?")

        assert "add function" in answer.lower()
        assert "Sources:" in answer
        assert "utils.py" in answer

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_ask_error_handling(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test error handling in ask method."""
        from repoqa.pipeline.rag import RAGPipeline

        # Mock vectorstore to raise an exception
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.side_effect = Exception("Database error")
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("test question")

        # When retrieval fails, pipeline still returns a response
        # (just without context)
        assert isinstance(answer, str)
        assert len(answer) > 0

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_index_repository(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_code_chunks,
        tmp_path,
    ):
        """Test repository indexing."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        mock_indexer.index_repository.return_value = {
            "chunks": sample_code_chunks,
            "embeddings": [[0.1] * 384] * len(sample_code_chunks),
            "file_count": 3,
            "repo_info": {},
            "repo_path": str(tmp_path / "repo"),
        }

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        result = pipeline.index_repository("test-repo")

        assert result["status"] == "success"
        assert result["documents_added"] == 3
        assert result["chunks_processed"] == 3
        mock_vectorstore.add_documents.assert_called_once()

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_safe_retriever_error_handling(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test safe retriever error handling."""
        from repoqa.pipeline.rag import RAGPipeline

        # Mock vectorstore to raise exception
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.side_effect = Exception("Search failed")
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        # Should return empty list on error
        docs = pipeline._safe_retriever("test query")
        assert docs == []

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_retrieve_and_format(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test retrieve and format method."""
        from repoqa.pipeline.rag import RAGPipeline

        # Mock vectorstore with valid documents
        valid_doc = Document(
            page_content="def test():\n    pass",
            metadata={"file_path": "test.py"},
        )
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [valid_doc]
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        result = pipeline._retrieve_and_format("test query")

        assert "test.py" in result
        assert "def test()" in result
        assert "```" in result

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_format_docs_with_integer_type(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test formatting documents with non-string content."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        # Test with invalid type (integer) using Mock
        doc = Mock()
        doc.page_content = 123  # Invalid type
        doc.metadata = {"file_path": "test.py"}
        docs = [doc]

        formatted = pipeline._format_docs(docs)
        assert "No valid documents found" in formatted

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_ask_with_empty_response(
        self,
        mock_embeddings,
        mock_chroma,
        tmp_path,
    ):
        """Test asking when LLM returns empty response."""
        from langchain_core.language_models import FakeListLLM

        from repoqa.pipeline.rag import RAGPipeline

        # Mock vectorstore
        valid_doc = Document(
            page_content="def test():\n    pass",
            metadata={"file_path": "test.py"},
        )
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = [valid_doc]
        mock_chroma.return_value = mock_vectorstore

        # Use FakeListLLM with empty response
        fake_llm = FakeListLLM(responses=[""])

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=fake_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("test question")

        assert "couldn't generate" in answer

    @patch("repoqa.pipeline.rag.Chroma")
    @patch("repoqa.pipeline.rag.HuggingFaceEmbeddings")
    def test_clean_response_multiline_think_tags(
        self,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test cleaning response with multiline think tags."""
        from repoqa.pipeline.rag import RAGPipeline

        mock_indexer = Mock()

        pipeline = RAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_indexer=mock_indexer,
        )

        # Test with multiline think tags
        response = """<think>
Let me analyze this...
Looking at the code...
</think>
The function does X."""
        cleaned = pipeline._clean_response(response)
        assert "The function does X." == cleaned
        assert "analyze" not in cleaned
        assert "<think>" not in cleaned
