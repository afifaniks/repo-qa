"""Sentence Transformer based embedding model implementation."""

from typing import List, Optional, Union

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from repoqa.embedding import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    """Embedding model implementation using Sentence Transformers.

    This implementation uses the SentenceTransformers library which provides
    state-of-the-art text embeddings. By default, it uses the 'all-MiniLM-L6-v2'
    model which is optimized for semantic similarity tasks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """Initialize the Sentence Transformer model.

        Args:
            model_name: Name or path of the model to load.
                      Defaults to 'all-MiniLM-L6-v2'.
            device: Device to run the model on ('cpu', 'cuda', etc.).
                   If None, automatically selects available device.
        """
        super().__init__(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.debug(
            f"Loading SentenceTransformer model '{model_name}' on device '{device}'"
        )
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model '{model_name}' loaded successfully.")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts to encode.
            **kwargs: Additional arguments passed to sentence_transformers.

        Returns:
            List of embeddings as float lists.
        """
        embeddings = self.model.encode(
            texts, convert_to_tensor=False, normalize_embeddings=True, **kwargs
        )

        # Handle single text input
        if isinstance(texts, str):
            return [embeddings.tolist()]

        return embeddings.tolist()

    def encode_batch(
        self, texts: List[str], batch_size: int = 32, **kwargs
    ) -> List[List[float]]:
        """Encode a large batch of texts efficiently.

        Args:
            texts: List of texts to encode.
            batch_size: Number of texts to encode at once.
            **kwargs: Additional arguments passed to sentence_transformers.

        Returns:
            List of embeddings as float lists.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True,
            **kwargs,
        )
        return embeddings.tolist()

    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimensionality of the embeddings.

        Returns:
            Integer dimension of the embedding vectors.
        """
        return self.model.get_sentence_embedding_dimension()
