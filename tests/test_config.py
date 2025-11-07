# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for configuration module."""

import os
from pathlib import Path

import pytest
import yaml


class TestConfig:
    """Test suite for Config class."""

    def test_config_loads_defaults(self, tmp_path):
        """Test that config loads default values correctly."""
        from repoqa.config import Config

        # Create a minimal config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# Empty config\n")

        config = Config(config_path=str(config_file))

        # Should have loaded from the default config.yaml
        assert config.llm_model is not None
        assert config.llm_backend is not None
        assert config.embedding_model is not None

    def test_config_override_with_user_values(self, tmp_path):
        """Test that user config overrides defaults."""
        from repoqa.config import Config

        # Create a user config with override
        user_config = {
            "llm": {
                "model": "custom-model",
                "temperature": 0.7,
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(user_config, f)

        config = Config(config_path=str(config_file))

        assert config.llm_model == "custom-model"
        assert config.llm_temperature == 0.7

    def test_config_get_method(self, sample_config_yaml):
        """Test the get method with dot notation."""
        from repoqa.config import Config

        config = Config(config_path=str(sample_config_yaml))

        assert config.get("llm.model") == "qwen3:1.7b"
        assert config.get("llm.temperature") == 0.3
        assert config.get("embedding.model") == "all-MiniLM-L6-v2"
        assert config.get("nonexistent.key", "default") == "default"

    def test_config_get_with_env_override(self, sample_config_yaml):
        """Test that environment variables override config values."""
        from repoqa.config import Config

        os.environ["LLM_MODEL"] = "env-model"
        os.environ["LLM_TEMPERATURE"] = "0.9"

        config = Config(config_path=str(sample_config_yaml))

        assert config.get("llm.model") == "env-model"
        assert config.get("llm.temperature") == 0.9

    def test_config_properties(self, sample_config_yaml):
        """Test all config property accessors."""
        from repoqa.config import Config

        config = Config(config_path=str(sample_config_yaml))

        # LLM properties
        assert config.llm_model == "qwen3:1.7b"
        assert config.llm_backend == "ollama"
        assert config.llm_temperature == 0.3
        assert config.ollama_base_url == "http://localhost:11434"

        # Embedding properties
        assert config.embedding_model == "all-MiniLM-L6-v2"

        # Vectorstore properties
        assert config.vectorstore_persist_directory == "./chroma_data"
        assert config.vectorstore_collection_prefix == "repoqa"
        assert config.vectorstore_chunk_size == 1024

        # Repository properties
        assert config.repository_clone_directory == "./repo_data"

        # Pipeline properties
        assert config.pipeline_mode == "agent"
        assert config.pipeline_max_iterations == 50
        assert config.pipeline_max_execution_time == 1000

        # API properties
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.api_title == "RepoQA API"
        assert config.api_description == "Repository Question Answering System"
        assert config.api_version == "0.1.0"

    def test_config_missing_file_uses_defaults(self, tmp_path):
        """Test that missing config file falls back to defaults."""
        from repoqa.config import Config

        nonexistent_file = tmp_path / "nonexistent.yaml"
        config = Config(config_path=str(nonexistent_file))

        # Should still work with defaults
        assert config.llm_model is not None
        assert config.embedding_model is not None

    def test_config_deep_merge(self, tmp_path):
        """Test deep merging of nested configurations."""
        from repoqa.config import Config

        # Create a config that partially overrides nested values
        partial_config = {
            "llm": {
                "model": "new-model",
                # temperature not specified, should use default
            },
            "api": {
                "port": 9000,
                # other api settings should use defaults
            },
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(partial_config, f)

        config = Config(config_path=str(config_file))

        # Check merged values
        assert config.llm_model == "new-model"
        assert config.llm_temperature is not None  # Should have default
        assert config.api_port == 9000
        assert config.api_host is not None  # Should have default

    def test_config_boolean_env_conversion(self, sample_config_yaml):
        """Test that boolean environment variables are converted correctly."""
        from repoqa.config import Config

        # Add a boolean to the config
        with open(sample_config_yaml, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict["test_bool"] = True

        with open(sample_config_yaml, "w") as f:
            yaml.dump(config_dict, f)

        # Test various boolean string representations
        os.environ["TEST_BOOL"] = "true"
        config = Config(config_path=str(sample_config_yaml))
        assert config.get("test_bool") is True

        os.environ["TEST_BOOL"] = "false"
        config = Config(config_path=str(sample_config_yaml))
        assert config.get("test_bool") is False

        os.environ["TEST_BOOL"] = "1"
        config = Config(config_path=str(sample_config_yaml))
        assert config.get("test_bool") is True

    def test_config_numeric_env_conversion(self, sample_config_yaml):
        """Test that numeric environment variables are converted correctly."""
        from repoqa.config import Config

        os.environ["VECTORSTORE_CHUNK_SIZE"] = "2048"
        os.environ["LLM_TEMPERATURE"] = "0.8"

        config = Config(config_path=str(sample_config_yaml))

        assert config.get("vectorstore.chunk_size") == 2048
        assert isinstance(config.get("vectorstore.chunk_size"), int)
        assert config.get("llm.temperature") == 0.8
        assert isinstance(config.get("llm.temperature"), float)
