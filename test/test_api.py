"""
Integration tests for the Normalized Semantic Chunker API.

Requires a running vLLM embedding endpoint (configured via EMBEDDING_BASE_URL
and EMBEDDING_API_KEY environment variables). The app no longer bundles a local
embedding model — all embeddings are fetched from the remote vLLM service.

Typical test invocation:
    EMBEDDING_BASE_URL=http://192.168.1.1:8000/v1 \
    EMBEDDING_API_KEY=<key> \
    pytest test/ -v
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from normalized_semantic_chunker import app

# Directory for test data files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Create test data directory and ensure test file exists."""
    # Create test data directory if it doesn't exist
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.txt")

    # Check if test file exists, if not raise an error
    if not os.path.exists(alice_path):
        raise FileNotFoundError(
            f"Required test file not found: {alice_path}. Please ensure the Alice in Wonderland text file exists in the {TEST_DATA_DIR} directory."
        )

    yield  # Run the tests


@pytest.fixture
def client():
    """Create a test client with actual embedder."""
    with TestClient(app) as test_client:
        yield test_client


def test_alice_file_processing(client):
    """Test processing alice_in_wonderland.txt and validate response structure."""
    # Path to the test file
    alice_path = os.path.join(TEST_DATA_DIR, "alice_in_wonderland.txt")

    # Open the file for sending to the API
    with open(alice_path, "rb") as f:
        # Send request to the API with the same parameters as request.py
        response = client.post(
            "/normalized_semantic_chunker/",
            files={"file": ("alice_in_wonderland.txt", f, "text/plain")},
            params={"max_tokens": 500},
        )

    # Check response status code
    assert response.status_code == 200, f"API returned error: {response.text}"

    # Parse the response
    data = response.json()

    # Basic structure validation
    assert "chunks" in data, "Response missing 'chunks' key"
    assert "metadata" in data, "Response missing 'metadata' key"

    # Validate chunks structure
    chunks = data["chunks"]
    assert isinstance(chunks, list), "'chunks' should be a list"
    assert len(chunks) > 0, "No chunks were generated"

    # Validate metadata structure
    metadata = data["metadata"]

    # Check required metadata fields
    required_metadata_fields = [
        "n_chunks",
        "avg_tokens",
        "max_tokens",
        "min_tokens",
        "percentile",
        "embedder_model",
        "source",
        "processing_time",
    ]

    for field in required_metadata_fields:
        assert field in metadata, f"Missing required metadata field: {field}"

    # Validate the source field value
    assert (
        metadata["source"] == "alice_in_wonderland.txt"
    ), f"Source field incorrect: expected 'alice_in_wonderland.txt', got '{metadata['source']}'"

    # Validate chunks structure
    for chunk in chunks:
        assert "text" in chunk, "Chunk missing 'text' field"
        assert "token_count" in chunk, "Chunk missing 'token_count' field"
        assert "id" in chunk, "Chunk missing 'id' field"

        # Validate types
        assert isinstance(chunk["text"], str), "Chunk text should be string"
        assert isinstance(chunk["token_count"], int), "Chunk token_count should be int"
        assert isinstance(chunk["id"], int), "Chunk id should be int"

        # Validate values
        assert len(chunk["text"]) > 0, "Chunk text should not be empty"
        assert chunk["token_count"] > 0, "Chunk token_count should be positive"
        assert chunk["id"] > 0, "Chunk id should be positive"

    # Log some information for debugging
    print(f"Successfully processed {len(chunks)} chunks")
    print(
        f"Token count range: {metadata['min_tokens']}-{metadata['max_tokens']} (avg: {metadata['avg_tokens']})"
    )
    print(f"Model used: {metadata['embedder_model']}")

    # Validate that the number of chunks matches the metadata
    assert len(chunks) == metadata["n_chunks"], (
        f"Number of chunks ({len(chunks)}) doesn't match metadata.n_chunks ({metadata['n_chunks']})"
    )

    # Validate that token counts are within the specified max_tokens
    for chunk in chunks:
        assert chunk["token_count"] <= 500, (
            f"Chunk {chunk['id']} exceeds max_tokens: {chunk['token_count']} > 500"
        )
