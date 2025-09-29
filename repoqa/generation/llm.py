# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Any, Dict

import torch
from transformers import pipeline


class HFLLMPipeline:
    """Generates answers using Hugging Face chat models."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the generator.

        Args:
            model_name: Name or path of the model to use.
            device: Device to run the model on.
        """
        self.chat = pipeline(
            "text2text-generation",
            model=model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
        )

    def generate_answer(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an answer based on query and context.

        Args:
            query: User's question.
            context: Retrieved code context.

        Returns:
            Generated answer.
        """
        # Format context
        context_str = context  # self._format_context(context.get("matches", []))

        # Build prompt
        prompt = (
            "System: You are a code assistant. Use the snippets to answer "
            "questions clearly and concisely. If uncertain about any details, "
            "acknowledge your uncertainty.\n\n"
            f"Code context:\n{context_str}\n\n"
            f"Question: {query}"
        )

        # Generate response
        response = self.chat(prompt, do_sample=True, temperature=0.7)
        return response[0]["generated_text"].strip()
