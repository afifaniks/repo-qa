import hashlib
import re
import socket
import sys
from urllib.parse import urlparse


def setup():
    host_name = socket.gethostname()

    if host_name == "hestia":
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


def get_collection_name(repo_url: str) -> str:
    """Generate a unique collection name from repository URL.

    Args:
        repo_url: Repository URL or path

    Returns:
        Sanitized collection name based on repo URL
    """
    # Parse URL to get repo name
    parsed = urlparse(repo_url)
    if parsed.netloc:  # It's a URL
        # Extract owner/repo from path
        path_parts = parsed.path.strip("/").replace(".git", "").split("/")
        if len(path_parts) >= 2:
            collection = f"{path_parts[-2]}_{path_parts[-1]}"
        else:
            collection = path_parts[-1] if path_parts else "unknown"
    else:  # It's a local path
        # Use the directory name
        collection = repo_url.strip("/").split("/")[-1]

    # Sanitize: replace invalid characters with underscore
    collection = re.sub(r"[^a-zA-Z0-9_-]", "_", collection)

    # ChromaDB collection names must be 3-63 chars, start/end with alphanumeric
    if len(collection) < 3:
        # Add hash suffix if too short
        hash_suffix = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        collection = f"{collection}_{hash_suffix}"
    elif len(collection) > 63:
        # Truncate and add hash if too long
        hash_suffix = hashlib.md5(repo_url.encode()).hexdigest()[:8]
        collection = f"{collection[:50]}_{hash_suffix}"

    # Ensure it starts and ends with alphanumeric
    collection = re.sub(r"^[^a-zA-Z0-9]+", "", collection)
    collection = re.sub(r"[^a-zA-Z0-9]+$", "", collection)

    return collection.lower()
