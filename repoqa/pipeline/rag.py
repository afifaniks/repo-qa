# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union


class RAGPipeline(ABC):
    @abstractmethod
    def index_repository(
        self,
        repo_path: Union[str, Path],
        clone_dir: Optional[str] = None,
    ) -> None:
        """Index the repository located at repo_path."""
        raise NotImplementedError()

    @abstractmethod
    def ask(self, query: str) -> str:
        """Ask a question about the indexed repository."""
        raise NotImplementedError()
