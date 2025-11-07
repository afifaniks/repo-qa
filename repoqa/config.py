# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


class Config:
    """Configuration manager for RepoQA."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._load_defaults()
        self._load_config()

    def _load_defaults(self):
        """Load default configuration from default config file."""
        # Try to load from config.yaml in the package directory
        default_config_path = Path(__file__).parent.parent / "config.yaml"

        if not default_config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {default_config_path}"
            )

        try:
            with open(default_config_path, "r") as f:
                self._defaults = yaml.safe_load(f) or {}
            logger.info(f"Loaded defaults from {default_config_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading default config: {e}")

    def _load_config(self):
        """Load configuration from YAML file."""
        # Start with defaults
        self._config = self._defaults.copy()

        if not self.config_path.exists():
            logger.info(f"Config file not found: {self.config_path}, " "using defaults")
            return

        try:
            with open(self.config_path, "r") as f:
                user_config = yaml.safe_load(f) or {}

            # Deep merge user config with defaults
            self._config = self._deep_merge(self._defaults, user_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            self._config = self._defaults.copy()

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with values to override

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with support for nested keys.

        Args:
            key: Configuration key (supports dot notation, e.g., 'llm.model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        # Override with environment variable if available
        env_key = "_".join(keys).upper()
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to convert to appropriate type
            if isinstance(value, bool):
                return env_value.lower() in ("true", "1", "yes")
            elif isinstance(value, int):
                return int(env_value)
            elif isinstance(value, float):
                return float(env_value)
            return env_value

        return value

    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return self.get("llm.model")

    @property
    def llm_backend(self) -> str:
        """Get LLM backend."""
        return self.get("llm.backend")

    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.get("llm.temperature")

    @property
    def ollama_base_url(self) -> str:
        """Get Ollama base URL."""
        return self.get("llm.ollama_base_url")

    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        return self.get("embedding.model")

    @property
    def vectorstore_persist_directory(self) -> str:
        """Get vector store persist directory."""
        return self.get("vectorstore.persist_directory")

    @property
    def vectorstore_collection_prefix(self) -> str:
        """Get vector store collection name prefix."""
        return self.get("vectorstore.collection_name_prefix")

    @property
    def repository_clone_directory(self) -> str:
        """Get repository clone directory."""
        return self.get("repository.clone_directory")

    @property
    def pipeline_mode(self) -> str:
        """Get pipeline mode."""
        return self.get("pipeline.mode")

    @property
    def pipeline_max_iterations(self) -> int:
        """Get pipeline max iterations."""
        return self.get("pipeline.max_iterations")

    @property
    def pipeline_max_execution_time(self) -> int:
        """Get pipeline max execution time."""
        return self.get("pipeline.max_execution_time")

    @property
    def api_host(self) -> str:
        """Get API host."""
        return self.get("api.host")

    @property
    def api_port(self) -> int:
        """Get API port."""
        return self.get("api.port")

    @property
    def api_title(self) -> str:
        """Get API title."""
        return self.get("api.title")

    @property
    def api_description(self) -> str:
        """Get API description."""
        return self.get("api.description")

    @property
    def api_version(self) -> str:
        """Get API version."""
        return self.get("api.version")


# Global config instance
config = Config()
