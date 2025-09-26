from abc import ABC, abstractmethod
from typing import List, Optional, Union


class EmbeddingModel(ABC):
    """Abstract base class for text embedding models.

    This class defines the interface that all embedding models must implement.
    Embedding models are used to convert text into dense vector representations
    that capture semantic meaning.
    """

    def __init__(self, model_name: str):
        """Initialize the embedding model.

        Args:
            model_name: Name or path of the embedding model to load.
        """
        self.model_name = model_name

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> List[List[float]]:
        """Encode text(s) into embeddings.

        Args:
            texts: Single text string or list of texts to encode.
            **kwargs: Additional arguments for the encoding process.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
            For a single input text, returns a list with one embedding.
            For multiple input texts, returns a list of embeddings.
        """
        raise NotImplementedError()

    @abstractmethod
    def encode_batch(
        self, texts: List[str], batch_size: int = 32, **kwargs
    ) -> List[List[float]]:
        """Encode a large batch of texts efficiently.

        Args:
            texts: List of texts to encode.
            batch_size: Number of texts to encode at once.
            **kwargs: Additional arguments for the encoding process.

        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimensionality of the embeddings.

        Returns:
            Integer dimension of the embedding vectors.
        """
        raise NotImplementedError()
