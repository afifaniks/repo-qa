# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Any, Dict

from repoqa.generation.llm import HFLLMPipeline


class AnswerGenerator:
    """Generates answers using an LLM."""

    def __init__(self, model_name: str):
        """Initialize the answer generator.

        Args:
            model_name: Name of the LLM to use.
        """
        self.model = HFLLMPipeline(model_name)

    def generate_answer(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an answer for a query using retrieved context.

        Args:
            query: User's natural language query.
            context: Retrieved code snippets and metadata.

        Returns:
            Generated answer string.
        """
        return self.model.generate_answer(query, context)
