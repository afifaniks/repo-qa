# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for Agentic RAG pipeline."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies():
    """Auto-mock dependencies for all tests in this module."""
    with patch("repoqa.pipeline.agentic_rag.SentenceTransformerEmbedding"), patch(
        "repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings"
    ), patch("repoqa.pipeline.agentic_rag.Chroma"), patch(
        "repoqa.pipeline.agentic_rag.create_react_agent"
    ), patch(
        "repoqa.pipeline.agentic_rag.AgentExecutor"
    ):
        yield


class TestAgenticRAGPipeline:
    """Test suite for AgenticRAGPipeline."""

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_initialization(
        self,
        mock_executor,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test agentic RAG pipeline initialization."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        assert pipeline.embedding_model_name == "test-model"
        assert pipeline.collection_name == "test-collection"
        assert pipeline.repo_path == repo_path
        assert len(pipeline.tools) == 4  # Should have 4 tools
        assert pipeline.accessed_files == set()

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_semantic_search_tool(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test semantic search tool."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock vectorstore
        mock_vectorstore = Mock()
        docs = [
            Document(
                page_content="def test():\n    pass",
                metadata={"file_path": "test.py"},
            )
        ]
        mock_vectorstore.similarity_search.return_value = docs
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        # Find and test semantic_search tool
        semantic_search_tool = next(
            t for t in pipeline.tools if t.name == "semantic_search"
        )
        result = semantic_search_tool.func("test query")

        assert "test.py" in result
        assert "def test()" in result
        assert "test.py" in pipeline.accessed_files

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_list_directory_tool(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test list directory tool."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        # Find list_directory tool
        list_dir_tool = next(t for t in pipeline.tools if t.name == "list_directory")

        # Test listing root directory
        result = list_dir_tool.func("")

        assert "README.md" in result
        assert "LICENSE" in result
        assert "src" in result or "[DIR]" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_read_file_tool(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test read file tool."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        # Find read_file tool
        read_file_tool = next(t for t in pipeline.tools if t.name == "read_file")

        # Test reading a file
        result = read_file_tool.func("README.md")

        assert "Test Repository" in result
        assert "README.md" in pipeline.accessed_files

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_read_file_nonexistent(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test reading non-existent file."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        read_file_tool = next(t for t in pipeline.tools if t.name == "read_file")

        result = read_file_tool.func("nonexistent.py")

        assert "does not exist" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_ask(
        self,
        mock_executor_class,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test asking a question."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        # Mock agent executor to simulate tool usage
        def mock_invoke(inputs):
            # Simulate tools accessing files
            pipeline.accessed_files.add("test.py")
            return {"output": "This is the agent's response."}

        mock_executor = Mock()
        mock_executor.invoke = mock_invoke
        mock_executor_class.return_value = mock_executor

        # Recreate the executor with our mock
        pipeline.agent_executor = mock_executor

        answer = pipeline.ask("What is this repository about?")

        assert "agent's response" in answer
        assert "Sources:" in answer
        assert "test.py" in answer

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_ask_error_handling(
        self,
        mock_executor_class,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test error handling in ask method."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock agent executor to raise exception
        mock_executor = Mock()
        mock_executor.invoke.side_effect = Exception("Agent error")
        mock_executor_class.return_value = mock_executor

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("test question")

        assert "Error" in answer

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_similarity_search_with_score_tool(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test similarity search with score tool."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock vectorstore
        mock_vectorstore = Mock()
        doc = Document(
            page_content="sample code",
            metadata={"file_path": "test.py"},
        )
        mock_vectorstore.similarity_search_with_score.return_value = [(doc, 0.85)]
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        # Find similarity_search_with_score tool
        tool = next(
            t for t in pipeline.tools if t.name == "similarity_search_with_score"
        )
        result = tool.func("test query")

        assert "0.85" in result or "0.850" in result
        assert "test.py" in result
        assert "sample code" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_semantic_search_no_results(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test semantic search with no results."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock vectorstore to return empty results
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.return_value = []
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        semantic_search_tool = next(
            t for t in pipeline.tools if t.name == "semantic_search"
        )
        result = semantic_search_tool.func("nonexistent query")

        assert "No relevant documents found" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_semantic_search_error(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test semantic search error handling."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock vectorstore to raise exception
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search.side_effect = Exception("DB error")
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        semantic_search_tool = next(
            t for t in pipeline.tools if t.name == "semantic_search"
        )
        result = semantic_search_tool.func("test query")

        assert "failed" in result.lower()

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_list_directory_nonexistent(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test listing non-existent directory."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        list_dir_tool = next(t for t in pipeline.tools if t.name == "list_directory")
        result = list_dir_tool.func("nonexistent/path")

        assert "does not exist" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_list_directory_file_path(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test listing directory when path is a file."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        list_dir_tool = next(t for t in pipeline.tools if t.name == "list_directory")
        result = list_dir_tool.func("README.md")

        assert "not a directory" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_list_directory_with_subdirectory(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test listing a subdirectory."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        list_dir_tool = next(t for t in pipeline.tools if t.name == "list_directory")
        result = list_dir_tool.func("src")

        assert "main.py" in result or "FILE" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_read_file_too_large(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test reading a file that's too large."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        # Create a large file
        large_file = repo_path / "large.txt"
        large_file.write_text("x" * 200000)  # 200KB

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        read_file_tool = next(t for t in pipeline.tools if t.name == "read_file")
        result = read_file_tool.func("large.txt")

        assert "too large" in result.lower()

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_read_file_is_directory(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test reading when path is a directory."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        read_file_tool = next(t for t in pipeline.tools if t.name == "read_file")
        result = read_file_tool.func("src")

        assert "not a file" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_read_file_empty_path(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test reading with empty file path."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        read_file_tool = next(t for t in pipeline.tools if t.name == "read_file")
        result = read_file_tool.func("")

        assert "cannot be empty" in result

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_ask_repo_not_exists(
        self,
        mock_executor_class,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        tmp_path,
    ):
        """Test asking when repository doesn't exist."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_indexer = Mock()
        nonexistent_path = tmp_path / "nonexistent_repo"

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(nonexistent_path),
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("test question")

        assert "does not exist" in answer

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_ask_iteration_limit(
        self,
        mock_executor_class,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_repo_structure,
    ):
        """Test asking when agent hits iteration limit."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        # Mock agent executor to return iteration limit message
        mock_executor = Mock()
        mock_executor.invoke.return_value = {
            "output": "Agent stopped due to iteration limit reached."
        }
        mock_executor_class.return_value = mock_executor

        mock_indexer = Mock()

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(sample_repo_structure.parent),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(sample_repo_structure),
            repo_indexer=mock_indexer,
        )

        answer = pipeline.ask("complex question")

        assert "iteration limit" in answer.lower()

    @patch("repoqa.pipeline.agentic_rag.Chroma")
    @patch("repoqa.pipeline.agentic_rag.HuggingFaceEmbeddings")
    @patch("repoqa.pipeline.agentic_rag.create_react_agent")
    @patch("repoqa.pipeline.agentic_rag.AgentExecutor")
    def test_index_repository(
        self,
        mock_executor,
        mock_agent,
        mock_embeddings,
        mock_chroma,
        mock_llm,
        sample_code_chunks,
        tmp_path,
    ):
        """Test repository indexing for agentic pipeline."""
        from repoqa.pipeline.agentic_rag import AgenticRAGPipeline

        mock_vectorstore = Mock()
        mock_chroma.return_value = mock_vectorstore

        mock_indexer = Mock()
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()

        mock_indexer.index_repository.return_value = {
            "chunks": sample_code_chunks,
            "embeddings": [[0.1] * 384] * len(sample_code_chunks),
            "file_count": 3,
            "repo_info": {},
            "repo_path": str(repo_path),
        }

        pipeline = AgenticRAGPipeline(
            llm_model=mock_llm,
            embedding_model="test-model",
            persist_directory=str(tmp_path),
            collection_name="test-collection",
            ollama_base_url="http://localhost:11434",
            temperature=0.5,
            repo_path=str(repo_path),
            repo_indexer=mock_indexer,
        )

        result = pipeline.index_repository(str(repo_path))

        assert result["status"] == "success"
        assert result["documents_added"] == 3
        assert result["file_exploration_enabled"] is True
        mock_vectorstore.add_documents.assert_called_once()
