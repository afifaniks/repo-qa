# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Shared test fixtures and configuration."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Mock chromadb module globally for all tests
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


@pytest.fixture(autouse=True)
def reset_chromadb_mock():
    """Reset chromadb mock between tests."""
    # Reset the mock completely
    chromadb_mock.PersistentClient.reset_mock()
    chromadb_mock.PersistentClient.return_value = None
    yield
    chromadb_mock.PersistentClient.reset_mock()
    chromadb_mock.PersistentClient.return_value = None


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "llm": {
            "model": "qwen3:1.7b",
            "backend": "ollama",
            "temperature": 0.3,
            "ollama_base_url": "http://localhost:11434",
        },
        "embedding": {"model": "all-MiniLM-L6-v2"},
        "vectorstore": {
            "persist_directory": "./chroma_data",
            "collection_name_prefix": "repoqa",
            "chunk_size": 1024,
        },
        "repository": {"clone_directory": "./repo_data"},
        "pipeline": {
            "mode": "agent",
            "max_iterations": 50,
            "max_execution_time": 1000,
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "title": "RepoQA API",
            "description": "Repository Question Answering System",
            "version": "0.1.0",
        },
    }


@pytest.fixture
def sample_config_yaml(temp_dir, sample_config_dict):
    """Create a temporary YAML config file."""
    import yaml

    config_file = temp_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_file


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock = Mock()
    mock.encode.return_value = [[0.1] * 384]
    mock.encode_batch.return_value = [[0.1] * 384, [0.2] * 384]
    mock.get_embedding_dim.return_value = 384
    return mock


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock = MagicMock()
    mock.encode.return_value = [[0.1] * 384]
    mock.get_sentence_embedding_dimension.return_value = 384
    return mock


@pytest.fixture
def mock_vectorstore():
    """Mock vector store for testing."""
    from langchain_core.documents import Document

    mock = Mock()
    mock.similarity_search.return_value = [
        Document(
            page_content="Sample code content",
            metadata={"file_path": "sample.py"},
        )
    ]
    mock.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="Sample code content",
                metadata={"file_path": "sample.py"},
            ),
            0.85,
        )
    ]
    mock.add_documents.return_value = None
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM model for testing using LangChain's FakeListLLM."""
    from langchain_core.language_models import FakeListLLM

    # FakeListLLM cycles through responses
    return FakeListLLM(responses=["This is a test response."])


@pytest.fixture
def sample_repo_structure(temp_dir):
    """Create a sample repository structure for testing."""
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir()

    # Create some sample files
    (repo_dir / "README.md").write_text("# Test Repository\n\nThis is a test.")
    (repo_dir / "LICENSE").write_text("MIT License\n\nCopyright (c) 2025")

    # Create source directory
    src_dir = repo_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text(
        'def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n'
    )
    (src_dir / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b\n"
    )

    # Create test directory
    test_dir = repo_dir / "tests"
    test_dir.mkdir()
    (test_dir / "test_utils.py").write_text(
        "import pytest\n\ndef test_add():\n    assert True\n"
    )

    # Create .gitignore
    (repo_dir / ".gitignore").write_text("__pycache__/\n*.pyc\n")

    return repo_dir


@pytest.fixture
def sample_code_chunks():
    """Sample code chunks for testing."""
    from repoqa.indexing.git_indexer import CodeChunk

    return [
        CodeChunk(content="def hello():\n    print('Hello')\n", file_path="test1.py"),
        CodeChunk(
            content="def goodbye():\n    print('Goodbye')\n", file_path="test2.py"
        ),
        CodeChunk(content="import os\nimport sys\n", file_path="test3.py"),
    ]


@pytest.fixture
def mock_git_repo():
    """Mock git repository."""
    mock = Mock()
    mock.active_branch.name = "main"
    mock.head.commit.hexsha = "abc123def456"
    mock.remotes.origin.urls = iter(["https://github.com/test/repo.git"])
    return mock


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "models": [
            {"name": "qwen3:1.7b", "size": 1234567890},
            {"name": "llama3.2:3b", "size": 2345678901},
        ]
    }


@pytest.fixture(autouse=True)
def reset_env():
    """Reset environment variables between tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
