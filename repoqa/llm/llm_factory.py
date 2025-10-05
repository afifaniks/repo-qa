from typing import Any, Dict, Optional

from langchain_ollama import OllamaLLM
from loguru import logger


def get_llm(
    model_name: str, backend: str, kwargs: Optional[Dict[str, Any]] = None
) -> Any:
    """Factory function to get LLM model instance based on backend.

    Args:
        model_name: Name of the model (e.g., 'llama3.2:3b', 'codellama:7b').
        backend: Backend type ('ollama' or 'other').

    Returns:
        An instance of the specified LLM model.
    """
    if backend == "ollama":
        if kwargs is None:
            kwargs = {}
        base_url = kwargs.get("base_url", "http://localhost:11435")
        temperature = kwargs.get("temperature", 0.5)

        logger.debug(
            f"Initializing Ollama model: {model_name} at {base_url} with temperature {temperature}"
        )
        return OllamaLLM(model=model_name, base_url=base_url, temperature=temperature)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
