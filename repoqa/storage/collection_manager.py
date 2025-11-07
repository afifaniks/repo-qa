# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Collection management utilities for ChromaDB."""

import hashlib
import re
from urllib.parse import urlparse

from loguru import logger


def get_collection_name(repo_url: str) -> str:
    """Generate a unique collection name from repository URL.

    Args:
        repo_url: Repository URL or path

    Returns:
        Sanitized collection name based on repo URL with unique hash
    """
    # Create a unique hash from the full repo URL to ensure uniqueness
    url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]

    # Parse URL to get a human-readable repo name
    parsed = urlparse(repo_url)
    if parsed.netloc:  # It's a URL
        # Extract owner/repo from path
        path_parts = parsed.path.strip("/").replace(".git", "").split("/")
        if len(path_parts) >= 2:
            repo_name = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            repo_name = path_parts[-1] if path_parts else "unknown"
    else:  # It's a local path
        # Use the directory name
        repo_name = repo_url.strip("/").split("/")[-1]

    # Sanitize: replace invalid characters with underscore
    repo_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)

    # Ensure repo_name starts with alphanumeric
    repo_name = re.sub(r"^[^a-zA-Z0-9]+", "", repo_name)

    # Truncate if needed to fit with hash
    # 63 is ChromaDB max length, -1 for underscore separator
    max_name_length = 63 - len(url_hash) - 1
    if len(repo_name) > max_name_length:
        repo_name = repo_name[:max_name_length]

    # Ensure it ends with alphanumeric after truncation
    repo_name = re.sub(r"[^a-zA-Z0-9]+$", "", repo_name)

    # Combine name with hash for uniqueness
    collection = f"{repo_name}_{url_hash}".lower()

    # Final check: ensure it's at least 3 chars
    if len(collection) < 3:
        collection = f"repo_{url_hash}".lower()

    return collection


def collection_exists_and_has_documents(
    persist_directory: str, collection_name: str
) -> bool:
    """Check if a collection exists and has documents.

    Args:
        persist_directory: Directory where ChromaDB persists data.
        collection_name: Name of the collection to check.

    Returns:
        True if collection exists and has documents, False otherwise.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_directory)

        # Get list of collections
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' does not exist")
            return False

        # Get the collection and check if it has documents
        collection = client.get_collection(name=collection_name)
        count = collection.count()

        logger.info(f"Collection '{collection_name}' has {count} documents")
        return count > 0

    except Exception as e:
        logger.error(f"Error checking collection: {e}")
        return False


def delete_collection(persist_directory: str, collection_name: str) -> bool:
    """Delete a collection from ChromaDB.

    Args:
        persist_directory: Directory where ChromaDB persists data.
        collection_name: Name of the collection to delete.

    Returns:
        True if collection was deleted successfully, False otherwise.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_directory)

        # Check if collection exists
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' does not exist")
            return True  # Nothing to delete, consider it success

        # Delete the collection
        client.delete_collection(name=collection_name)
        logger.info(f"Successfully deleted collection '{collection_name}'")
        return True

    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        return False


def list_collections(persist_directory: str) -> list[str]:
    """List all collections in ChromaDB.

    Args:
        persist_directory: Directory where ChromaDB persists data.

    Returns:
        List of collection names.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_directory)
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        logger.info(f"Found {len(collection_names)} collections")
        return collection_names

    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []


def get_collection_info(persist_directory: str, collection_name: str) -> dict:
    """Get information about a collection.

    Args:
        persist_directory: Directory where ChromaDB persists data.
        collection_name: Name of the collection to get info for.

    Returns:
        Dictionary with collection information.
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_directory)

        # Check if collection exists
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            return {"exists": False, "name": collection_name}

        # Get collection info
        collection = client.get_collection(name=collection_name)
        count = collection.count()

        return {
            "exists": True,
            "name": collection_name,
            "document_count": count,
        }

    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        return {"exists": False, "name": collection_name, "error": str(e)}
