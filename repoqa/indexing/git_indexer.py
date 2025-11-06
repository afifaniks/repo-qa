# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import git
from loguru import logger
from tqdm import tqdm

from repoqa.embedding.embedding_model import EmbeddingModel
from repoqa.indexing.indexer import RepoIndexer


@dataclass
class CodeChunk:
    """Represents a chunk of code with minimal metadata."""

    content: str
    file_path: str


class GitRepoIndexer(RepoIndexer):
    """Repository indexer using git and simple text chunking."""

    IGNORE_PATTERNS = {
        "__pycache__",
        "node_modules",
        "venv",
        ".env",
        ".git",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
    }

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        chunk_size: int = 1024,
        batch_size: int = 32,
    ):
        super().__init__(embedding_model)
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def _should_ignore(self, path: str) -> bool:
        path_parts = Path(path).parts
        return any(
            ignore in path_parts or path.endswith(ignore)
            for ignore in self.IGNORE_PATTERNS
        )

    def _chunk_file(self, file_path: str) -> List[CodeChunk]:
        """Split a file into simple text chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return []

        chunks = []
        start = 0
        while start < len(lines):
            end = min(start + self.chunk_size, len(lines))
            chunk_text = "".join(lines[start:end])
            chunks.append(CodeChunk(content=chunk_text, file_path=file_path))
            start = end

        return chunks

    def _find_code_files(self, repo_path: str) -> Set[str]:
        code_files = set()
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not self._should_ignore(d)]
            for file in files:
                file_path = os.path.join(root, file)
                if not self._should_ignore(file_path):
                    code_files.add(file_path)
        return code_files

    def _is_git_url(self, repo_path: str) -> bool:
        git_prefixes = ("git@", "https://", "git://")
        return any(repo_path.startswith(prefix) for prefix in git_prefixes)

    def _clone_repository(self, repo_url: str, target_dir: str) -> str:
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            clone_path = os.path.join(target_dir, repo_name)
            if os.path.exists(clone_path):
                repo = git.Repo(clone_path)
                repo.remotes.origin.pull()
                return clone_path
            git.Repo.clone_from(repo_url, clone_path)
            return clone_path
        except git.GitCommandError as e:
            raise ValueError(f"Failed to clone repository: {str(e)}")

    def index_repository(
        self, repo_path: str, clone_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        import tempfile

        temp_dir = None
        try:
            if self._is_git_url(repo_path):
                if clone_dir is None:
                    temp_dir = tempfile.mkdtemp()
                    clone_dir = temp_dir
                repo_path = self._clone_repository(repo_path, clone_dir)

            code_files = self._find_code_files(repo_path)
            logger.debug(f"Found {len(code_files)} code files.")

            chunks = []
            with ThreadPoolExecutor() as executor:
                chunk_lists = list(
                    tqdm(
                        executor.map(self._chunk_file, code_files),
                        total=len(code_files),
                        desc="Chunking files",
                    )
                )
                chunks = [chunk for chunk_list in chunk_lists for chunk in chunk_list]

            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode_batch(
                texts, batch_size=self.batch_size
            )

            try:
                repo = git.Repo(repo_path)
                repo_info = {
                    "remote_url": next(repo.remotes.origin.urls, None),
                    "default_branch": repo.active_branch.name,
                    "commit_hash": repo.head.commit.hexsha,
                }
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                repo_info = {}

            return {
                "chunks": chunks,
                "embeddings": embeddings,
                "file_count": len(code_files),
                "repo_info": repo_info,
            }
        finally:
            if temp_dir:
                import shutil

                shutil.rmtree(temp_dir)
