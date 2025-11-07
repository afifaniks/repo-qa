# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from repoqa.app import RepoQA
from repoqa.llm.llm_factory import get_llm
from repoqa.util.setup_util import get_collection_name, setup

setup()

app = FastAPI(
    title="RepoQA API",
    description="Repository-level Question Answering with RAG",
    version="1.0.0",
)

# Cache for RepoQA instances per repository
repo_qa_cache = {}


class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    repo: str = Field(
        ...,
        description="Repository URL or path to analyze",
        example="https://github.com/afifaniks/repoqa.git",
    )
    question: str = Field(
        ..., description="Question to ask about the repository", min_length=1
    )
    llm_model: Optional[str] = Field(
        default="qwen3:1.7b", description="LLM model to use"
    )
    skip_indexing: Optional[bool] = Field(
        default=False,
        description="Skip indexing if repo already indexed",
    )


class AnswerResponse(BaseModel):
    """Response model for answers."""

    question: str
    answer: str
    repo: str
    indexed: bool


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RepoQA API",
        "version": "1.0.0",
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about a repository.

    Args:
        request: Question request with repo, question, and options

    Returns:
        Answer response with the generated answer
    """
    try:
        # Generate collection name for this repository
        collection_name = get_collection_name(request.repo)
        logger.info(f"Using collection '{collection_name}' for repo: {request.repo}")

        # Check if we have a cached instance for this repo
        if request.repo not in repo_qa_cache:
            logger.info(f"Initializing RepoQA for repo: {request.repo}")
            llm_model = request.llm_model or "qwen3:1.7b"
            repo_qa_cache[request.repo] = RepoQA(
                persist_directory="./chroma_data",
                collection_name=collection_name,
                llm_model=get_llm(llm_model, backend="ollama"),
                mode="agent",
            )

        repo_qa_instance = repo_qa_cache[request.repo]

        # Index repository if needed
        indexed = False
        if not request.skip_indexing:
            logger.info(f"Indexing repository: {request.repo}")
            result = repo_qa_instance.index_repository(
                repo_path=request.repo,
                clone_dir="./repo_data",
            )
            logger.info(f"Indexing completed: {result}")
            indexed = True

        # Ask question
        logger.info(f"Processing question: {request.question}")
        answer = repo_qa_instance.ask(request.question)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            repo=request.repo,
            indexed=indexed,
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "repos_loaded": len(repo_qa_cache),
        "loaded_repos": list(repo_qa_cache.keys()),
    }
