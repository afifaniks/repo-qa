# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for embedding models."""

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestSentenceTransformerEmbedding:
    """Test suite for SentenceTransformerEmbedding."""

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_initialization(self, mock_st):
        """Test embedding model initialization."""
        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding(model_name="test-model", device="cpu")

        assert embedding.model_name == "test-model"
        assert embedding.device == "cpu"
        mock_st.assert_called_once_with("test-model", device="cpu")

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    @patch("repoqa.embedding.sentence_transformer.torch")
    def test_device_selection(self, mock_torch, mock_st):
        """Test automatic device selection."""
        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        # Test CUDA available
        mock_torch.cuda.is_available.return_value = True
        mock_st.return_value = MagicMock()

        embedding = SentenceTransformerEmbedding(model_name="test-model")
        assert embedding.device == "cuda"

        # Test CUDA not available
        mock_torch.cuda.is_available.return_value = False
        embedding = SentenceTransformerEmbedding(model_name="test-model")
        assert embedding.device == "cpu"

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_single_text(self, mock_st):
        """Test encoding a single text."""
        import numpy as np

        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        # Return numpy array for single text
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        result = embedding.encode("test text")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once_with(
            "test text",
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_multiple_texts(self, mock_st):
        """Test encoding multiple texts."""
        import numpy as np

        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        texts = ["text1", "text2"]
        result = embedding.encode(texts)

        assert isinstance(result, list)
        assert len(result) == 2
        mock_model.encode.assert_called_once_with(
            texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_batch(self, mock_st):
        """Test batch encoding."""
        import numpy as np

        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        texts = ["text1", "text2", "text3"]
        result = embedding.encode_batch(texts, batch_size=2)

        assert isinstance(result, list)
        assert len(result) == 3
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=2,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_get_embedding_dim(self, mock_st):
        """Test getting embedding dimension."""
        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        dim = embedding.get_embedding_dim()

        assert dim == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()

    @patch("repoqa.embedding.sentence_transformer.SentenceTransformer")
    def test_encode_with_kwargs(self, mock_st):
        """Test encoding with additional kwargs."""
        import numpy as np

        from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding

        mock_model = MagicMock()
        # Return numpy array for single text
        mock_model.encode.return_value = np.array([0.1, 0.2])
        mock_st.return_value = mock_model

        embedding = SentenceTransformerEmbedding()

        result = embedding.encode(
            "test text",
            show_progress_bar=False,
            device="cpu",
        )

        assert isinstance(result, list)
        assert len(result) == 1
        mock_model.encode.assert_called_once_with(
            "test text",
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=False,
            device="cpu",
        )
