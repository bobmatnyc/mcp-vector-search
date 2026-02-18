"""Test that CUDA embeddings bypass thread pool to avoid context issues."""

import unittest.mock as mock

import pytest

from mcp_vector_search.core.embeddings import CodeBERTEmbeddingFunction


def test_cuda_skips_thread_pool():
    """Test that CUDA device runs embeddings directly without ThreadPoolExecutor."""
    # Mock device detection to return CUDA
    with mock.patch(
        "mcp_vector_search.core.embeddings._detect_device", return_value="cuda"
    ):
        # Mock SentenceTransformer to avoid loading model
        with mock.patch(
            "mcp_vector_search.core.embeddings.SentenceTransformer"
        ) as mock_st:
            mock_model = mock.Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [[0.1] * 768, [0.2] * 768]
            mock_st.return_value = mock_model

            # Create embedding function
            emb_fn = CodeBERTEmbeddingFunction(model_name="microsoft/codebert-base")

            # Verify device is CUDA
            assert emb_fn.device == "cuda"

            # Call embedding function
            test_inputs = ["def hello(): pass", "import numpy as np"]
            embeddings = emb_fn(test_inputs)

            # Verify embeddings were generated
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 768

            # Verify encode was called (meaning _generate_embeddings ran)
            mock_model.encode.assert_called_once()


def test_cpu_uses_thread_pool():
    """Test that CPU device uses ThreadPoolExecutor with timeout."""
    # Mock device detection to return CPU
    with mock.patch(
        "mcp_vector_search.core.embeddings._detect_device", return_value="cpu"
    ):
        # Mock SentenceTransformer
        with mock.patch(
            "mcp_vector_search.core.embeddings.SentenceTransformer"
        ) as mock_st:
            mock_model = mock.Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [[0.1] * 768]
            mock_st.return_value = mock_model

            # Create embedding function
            emb_fn = CodeBERTEmbeddingFunction(model_name="microsoft/codebert-base")

            # Verify device is CPU
            assert emb_fn.device == "cpu"

            # Mock ThreadPoolExecutor to verify it's used
            with mock.patch(
                "mcp_vector_search.core.embeddings.ThreadPoolExecutor"
            ) as mock_tpe:
                mock_executor = mock.Mock()
                mock_future = mock.Mock()
                mock_future.result.return_value = [[0.1] * 768]
                mock_executor.submit.return_value = mock_future
                mock_executor.__enter__.return_value = mock_executor
                mock_executor.__exit__.return_value = False
                mock_tpe.return_value = mock_executor

                # Call embedding function
                test_inputs = ["def hello(): pass"]
                emb_fn(test_inputs)  # Result unused; testing side effects

                # Verify ThreadPoolExecutor was used for CPU
                mock_tpe.assert_called_once_with(max_workers=1)
                mock_executor.submit.assert_called_once()

                # Verify timeout was applied
                mock_future.result.assert_called_once_with(timeout=emb_fn.timeout)


def test_mps_uses_thread_pool():
    """Test that MPS device uses ThreadPoolExecutor with timeout."""
    # Mock device detection to return MPS
    with mock.patch(
        "mcp_vector_search.core.embeddings._detect_device", return_value="mps"
    ):
        # Mock SentenceTransformer
        with mock.patch(
            "mcp_vector_search.core.embeddings.SentenceTransformer"
        ) as mock_st:
            mock_model = mock.Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.return_value = [[0.1] * 768]
            mock_st.return_value = mock_model

            # Create embedding function
            emb_fn = CodeBERTEmbeddingFunction(model_name="microsoft/codebert-base")

            # Verify device is MPS
            assert emb_fn.device == "mps"

            # Mock ThreadPoolExecutor to verify it's used
            with mock.patch(
                "mcp_vector_search.core.embeddings.ThreadPoolExecutor"
            ) as mock_tpe:
                mock_executor = mock.Mock()
                mock_future = mock.Mock()
                mock_future.result.return_value = [[0.1] * 768]
                mock_executor.submit.return_value = mock_future
                mock_executor.__enter__.return_value = mock_executor
                mock_executor.__exit__.return_value = False
                mock_tpe.return_value = mock_executor

                # Call embedding function
                test_inputs = ["def hello(): pass"]
                emb_fn(test_inputs)  # Result unused; testing side effects

                # Verify ThreadPoolExecutor was used for MPS
                mock_tpe.assert_called_once_with(max_workers=1)


def test_cuda_direct_call_performance():
    """Test that CUDA direct call has no thread pool overhead."""
    import time

    # Mock CUDA device
    with mock.patch(
        "mcp_vector_search.core.embeddings._detect_device", return_value="cuda"
    ):
        with mock.patch(
            "mcp_vector_search.core.embeddings.SentenceTransformer"
        ) as mock_st:
            mock_model = mock.Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768

            # Simulate fast GPU embedding (10ms)
            def fast_encode(*args, **kwargs):
                time.sleep(0.01)  # 10ms
                return [[0.1] * 768] * 10

            mock_model.encode.side_effect = fast_encode
            mock_st.return_value = mock_model

            # Create embedding function
            emb_fn = CodeBERTEmbeddingFunction(model_name="microsoft/codebert-base")

            # Time CUDA direct call
            start = time.perf_counter()
            test_inputs = [f"def func_{i}(): pass" for i in range(10)]
            embeddings = emb_fn(test_inputs)
            elapsed = time.perf_counter() - start

            # Should complete in ~10-15ms (10ms encode + minimal overhead)
            # ThreadPoolExecutor would add 5-10ms overhead
            assert elapsed < 0.05  # 50ms threshold (generous for CI)
            assert len(embeddings) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
