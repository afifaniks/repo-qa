# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for indexing module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestGitRepoIndexer:
    """Test suite for GitRepoIndexer."""

    def test_initialization(self, mock_embedding_model):
        """Test indexer initialization."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(
            embedding_model=mock_embedding_model,
            chunk_size=512,
            batch_size=16,
        )

        assert indexer.chunk_size == 512
        assert indexer.batch_size == 16
        assert indexer.embedding_model == mock_embedding_model

    def test_should_ignore_patterns(self, mock_embedding_model):
        """Test ignore pattern matching."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        # Should ignore
        assert indexer._should_ignore("__pycache__/file.py")
        assert indexer._should_ignore("node_modules/package.json")
        assert indexer._should_ignore(".git/config")
        assert indexer._should_ignore("file.pyc")
        assert indexer._should_ignore("venv/lib/python")

        # Should not ignore
        assert not indexer._should_ignore("src/main.py")
        assert not indexer._should_ignore("README.md")
        assert not indexer._should_ignore("tests/test_main.py")

    def test_chunk_file(self, mock_embedding_model, tmp_path):
        """Test file chunking."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(
            embedding_model=mock_embedding_model,
            chunk_size=3,  # Small chunk for testing
        )

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        chunks = indexer._chunk_file(str(test_file))

        assert len(chunks) == 2  # 5 lines with chunk_size=3 = 2 chunks
        assert chunks[0].file_path == str(test_file)
        assert chunks[1].file_path == str(test_file)
        assert "line1" in chunks[0].content
        assert "line4" in chunks[1].content

    def test_chunk_file_unicode_error(self, mock_embedding_model, tmp_path):
        """Test handling of files with encoding errors."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        # Create a binary file that will cause UnicodeDecodeError
        test_file = tmp_path / "binary.dat"
        test_file.write_bytes(b"\x80\x81\x82\x83")

        chunks = indexer._chunk_file(str(test_file))

        assert len(chunks) == 0  # Should return empty list

    def test_find_code_files(self, mock_embedding_model, sample_repo_structure):
        """Test finding code files in a repository."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        files = indexer._find_code_files(str(sample_repo_structure))

        # Should find non-ignored files
        file_names = [Path(f).name for f in files]
        assert "README.md" in file_names
        assert "LICENSE" in file_names
        assert "main.py" in file_names
        assert "utils.py" in file_names

        # Should not find ignored files
        assert ".gitignore" in file_names  # .gitignore itself is found
        # But no __pycache__ or .pyc files

    def test_is_git_url(self, mock_embedding_model):
        """Test git URL detection."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        # Valid git URLs
        assert indexer._is_git_url("https://github.com/user/repo.git")
        assert indexer._is_git_url("git@github.com:user/repo.git")
        assert indexer._is_git_url("git://github.com/user/repo.git")

        # Not git URLs
        assert not indexer._is_git_url("/local/path/to/repo")
        assert not indexer._is_git_url("./relative/path")
        assert not indexer._is_git_url("C:\\Windows\\Path")

    @patch("repoqa.indexing.git_indexer.git.Repo")
    def test_clone_repository_new(
        self, mock_repo_class, mock_embedding_model, tmp_path
    ):
        """Test cloning a new repository."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        repo_url = "https://github.com/user/test-repo.git"
        target_dir = str(tmp_path)

        # Mock the clone operation
        mock_repo_class.clone_from.return_value = MagicMock()

        result = indexer._clone_repository(repo_url, target_dir)

        expected_path = str(tmp_path / "test-repo")
        assert result == expected_path
        mock_repo_class.clone_from.assert_called_once_with(repo_url, expected_path)

    @patch("repoqa.indexing.git_indexer.git.Repo")
    def test_clone_repository_existing(
        self, mock_repo_class, mock_embedding_model, tmp_path
    ):
        """Test updating an existing cloned repository."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        # Create existing directory
        repo_dir = tmp_path / "test-repo"
        repo_dir.mkdir()

        repo_url = "https://github.com/user/test-repo.git"
        target_dir = str(tmp_path)

        # Mock the repo and pull operation
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        result = indexer._clone_repository(repo_url, target_dir)

        expected_path = str(repo_dir)
        assert result == expected_path
        mock_repo.remotes.origin.pull.assert_called_once()

    def test_index_repository_local(self, mock_embedding_model, sample_repo_structure):
        """Test indexing a local repository."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        mock_embedding_model.encode_batch.return_value = [[0.1] * 384] * 10

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        result = indexer.index_repository(repo_path=str(sample_repo_structure))

        # Check that result has expected keys
        assert "chunks" in result
        assert "embeddings" in result
        assert "file_count" in result
        assert result["file_count"] > 0
        assert len(result["chunks"]) > 0

    @patch("repoqa.indexing.git_indexer.git.Repo")
    def test_index_repository_with_git_info(
        self, mock_repo_class, mock_embedding_model, sample_repo_structure
    ):
        """Test indexing with git repository info."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        # Mock git repository
        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        mock_repo.head.commit.hexsha = "abc123"
        mock_repo.remotes.origin.urls = iter(["https://github.com/test/repo.git"])
        mock_repo_class.return_value = mock_repo

        mock_embedding_model.encode_batch.return_value = [[0.1] * 384]

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        result = indexer.index_repository(repo_path=str(sample_repo_structure))

        assert "repo_info" in result
        assert result["repo_info"]["default_branch"] == "main"
        assert result["repo_info"]["commit_hash"] == "abc123"

    @patch("repoqa.indexing.git_indexer.git.Repo")
    def test_index_repository_git_url(
        self, mock_repo_class, mock_embedding_model, tmp_path
    ):
        """Test indexing a repository from git URL."""
        from repoqa.indexing.git_indexer import GitRepoIndexer

        # Create a mock repository structure
        clone_dir = tmp_path / "test-repo"
        clone_dir.mkdir()
        (clone_dir / "README.md").write_text("# Test")

        # Mock the clone operation
        mock_repo_class.clone_from.return_value = MagicMock()

        mock_embedding_model.encode_batch.return_value = [[0.1] * 384]

        indexer = GitRepoIndexer(embedding_model=mock_embedding_model)

        # Mock _clone_repository to return our test directory
        with patch.object(indexer, "_clone_repository", return_value=str(clone_dir)):
            result = indexer.index_repository(
                repo_path="https://github.com/test/repo.git",
                clone_dir=str(tmp_path),
            )

        assert "chunks" in result
        assert result["repo_path"] == str(clone_dir)
