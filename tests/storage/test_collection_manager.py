# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Tests for collection management utilities."""

import sys
from unittest.mock import MagicMock, Mock

import pytest

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


@pytest.fixture(autouse=True)
def reset_chromadb_mock():
    """Reset chromadb mock between tests."""
    chromadb_mock.PersistentClient.reset_mock()
    chromadb_mock.PersistentClient.return_value = None
    yield
    chromadb_mock.PersistentClient.reset_mock()
    chromadb_mock.PersistentClient.return_value = None


class TestGetCollectionName:
    """Test suite for get_collection_name function."""

    def test_github_url_with_owner_repo(self):
        """Test generating collection name from GitHub URL."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("https://github.com/owner/repo.git")

        assert result.startswith("owner_repo_")
        assert len(result) <= 63  # ChromaDB max length
        assert result.islower()
        assert len(result) >= 3

    def test_github_url_without_git_suffix(self):
        """Test GitHub URL without .git suffix."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("https://github.com/user/project")

        assert result.startswith("user_project_")
        assert len(result) <= 63

    def test_local_path(self):
        """Test generating collection name from local path."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("/path/to/my-repo")

        assert result.startswith("my-repo_")
        assert len(result) <= 63

    def test_url_with_special_characters(self):
        """Test URL with special characters are sanitized."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("https://github.com/user/my@repo#123")

        # Special characters should be replaced with underscores
        assert "@" not in result
        assert "#" not in result
        assert len(result) <= 63

    def test_very_long_repo_name(self):
        """Test that very long repo names are truncated."""
        from repoqa.storage.collection_manager import get_collection_name

        long_name = "a" * 100
        result = get_collection_name(f"https://github.com/owner/{long_name}")

        assert len(result) <= 63
        # Should still have the hash at the end
        assert "_" in result

    def test_url_with_subdirectories(self):
        """Test URL with subdirectories in path."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("https://github.com/org/repo/tree/branch/path")

        # Should use last two path components (branch and path)
        assert "branch_path_" in result
        assert len(result) <= 63

    def test_same_url_generates_same_name(self):
        """Test that the same URL always generates the same name."""
        from repoqa.storage.collection_manager import get_collection_name

        url = "https://github.com/test/repo.git"
        result1 = get_collection_name(url)
        result2 = get_collection_name(url)

        assert result1 == result2

    def test_different_urls_generate_different_names(self):
        """Test that different URLs generate different names."""
        from repoqa.storage.collection_manager import get_collection_name

        result1 = get_collection_name("https://github.com/user/repo1.git")
        result2 = get_collection_name("https://github.com/user/repo2.git")

        assert result1 != result2

    def test_collection_name_starts_with_alphanumeric(self):
        """Test that collection name starts with alphanumeric character."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("https://github.com/user/_-_repo")

        # Should strip leading non-alphanumeric
        assert result[0].isalnum()

    def test_collection_name_minimum_length(self):
        """Test that collection name meets minimum length requirement."""
        from repoqa.storage.collection_manager import get_collection_name

        result = get_collection_name("x")

        assert len(result) >= 3
        assert result.startswith("repo_") or len(result) >= 3


class TestCollectionManager:
    """Test suite for collection manager functions."""

    def test_collection_exists_and_has_documents(self):
        """Test checking if collection exists with documents."""
        from repoqa.storage.collection_manager import (
            collection_exists_and_has_documents,
        )

        # Mock collection
        mock_collection = Mock()
        mock_collection.count.return_value = 10
        mock_collection.name = "test-collection"

        # Mock client
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        # Set up the mock
        chromadb_mock.PersistentClient.return_value = mock_client

        result = collection_exists_and_has_documents("/path/to/db", "test-collection")

        assert result is True
        mock_client.get_collection.assert_called_once_with(name="test-collection")

    def test_collection_not_exists(self):
        """Test checking non-existent collection."""
        from repoqa.storage.collection_manager import (
            collection_exists_and_has_documents,
        )

        # Mock client with no collections
        mock_client = Mock()
        mock_client.list_collections.return_value = []

        chromadb_mock.PersistentClient.return_value = mock_client

        result = collection_exists_and_has_documents("/path/to/db", "nonexistent")

        assert result is False

    def test_collection_exists_empty(self):
        """Test checking collection with no documents."""
        from repoqa.storage.collection_manager import (
            collection_exists_and_has_documents,
        )

        # Mock empty collection
        mock_collection = Mock()
        mock_collection.count.return_value = 0
        mock_collection.name = "test-collection"

        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        chromadb_mock.PersistentClient.return_value = mock_client

        result = collection_exists_and_has_documents("/path/to/db", "test-collection")

        assert result is False

    def test_collection_exists_error_handling(self):
        """Test error handling when checking collection."""
        from repoqa.storage.collection_manager import (
            collection_exists_and_has_documents,
        )

        # Mock client that raises an error
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("Test error")

        chromadb_mock.PersistentClient.return_value = mock_client

        result = collection_exists_and_has_documents("/path/to/db", "test-collection")

        assert result is False

    def test_delete_collection(self):
        """Test deleting a collection."""
        from repoqa.storage.collection_manager import delete_collection

        # Mock collection
        mock_collection = Mock()
        mock_collection.name = "test-collection"

        # Mock client
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_collection]

        chromadb_mock.PersistentClient.return_value = mock_client

        result = delete_collection("/path/to/db", "test-collection")

        assert result is True
        mock_client.delete_collection.assert_called_once_with(name="test-collection")

    def test_delete_nonexistent_collection(self):
        """Test deleting a non-existent collection."""
        from repoqa.storage.collection_manager import delete_collection

        # Mock client with no collections
        mock_client = Mock()
        mock_client.list_collections.return_value = []

        chromadb_mock.PersistentClient.return_value = mock_client

        result = delete_collection("/path/to/db", "nonexistent")

        assert result is True
        mock_client.delete_collection.assert_not_called()

    def test_delete_collection_error_handling(self):
        """Test error handling when deleting collection."""
        from repoqa.storage.collection_manager import delete_collection

        # Mock client that raises an error
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("Test error")

        chromadb_mock.PersistentClient.return_value = mock_client

        result = delete_collection("/path/to/db", "test-collection")

        assert result is False

    def test_list_collections(self):
        """Test listing all collections."""
        from repoqa.storage.collection_manager import list_collections

        # Mock collections
        mock_col1 = Mock()
        mock_col1.name = "collection-1"
        mock_col2 = Mock()
        mock_col2.name = "collection-2"

        # Mock client
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_col1, mock_col2]

        chromadb_mock.PersistentClient.return_value = mock_client

        result = list_collections("/path/to/db")

        assert result == ["collection-1", "collection-2"]
        mock_client.list_collections.assert_called_once()

    def test_list_collections_empty(self):
        """Test listing collections when none exist."""
        from repoqa.storage.collection_manager import list_collections

        # Mock client with no collections
        mock_client = Mock()
        mock_client.list_collections.return_value = []

        chromadb_mock.PersistentClient.return_value = mock_client

        result = list_collections("/path/to/db")

        assert result == []

    def test_list_collections_error_handling(self):
        """Test error handling when listing collections."""
        from repoqa.storage.collection_manager import list_collections

        # Mock client that raises an error
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("Test error")

        chromadb_mock.PersistentClient.return_value = mock_client

        result = list_collections("/path/to/db")

        assert result == []

    def test_get_collection_info(self):
        """Test getting collection information."""
        from repoqa.storage.collection_manager import get_collection_info

        # Mock collection
        mock_collection = Mock()
        mock_collection.count.return_value = 25
        mock_collection.name = "test-collection"

        # Mock client
        mock_client = Mock()
        mock_client.list_collections.return_value = [mock_collection]
        mock_client.get_collection.return_value = mock_collection

        chromadb_mock.PersistentClient.return_value = mock_client

        result = get_collection_info("/path/to/db", "test-collection")

        assert result == {
            "exists": True,
            "name": "test-collection",
            "document_count": 25,
        }

    def test_get_collection_info_nonexistent(self):
        """Test getting info for non-existent collection."""
        from repoqa.storage.collection_manager import get_collection_info

        # Mock client with no collections
        mock_client = Mock()
        mock_client.list_collections.return_value = []

        chromadb_mock.PersistentClient.return_value = mock_client

        result = get_collection_info("/path/to/db", "nonexistent")

        assert result == {"exists": False, "name": "nonexistent"}

    def test_get_collection_info_error_handling(self):
        """Test error handling when getting collection info."""
        from repoqa.storage.collection_manager import get_collection_info

        # Mock client that raises an error
        mock_client = Mock()
        mock_client.list_collections.side_effect = Exception("Test error")

        chromadb_mock.PersistentClient.return_value = mock_client

        result = get_collection_info("/path/to/db", "test-collection")

        assert result["exists"] is False
        assert result["name"] == "test-collection"
        assert "error" in result
