"""Maximal Marginal Relevance (MMR) for diverse search results.

MMR balances relevance and diversity by reranking candidates to minimize
redundancy while maintaining semantic similarity to the query.

Algorithm:
    Score = λ * relevance(query, doc) - (1-λ) * max_similarity(doc, selected_docs)

Where:
    - λ (lambda_param): Balance between relevance (1.0) and diversity (0.0)
    - relevance(query, doc): Original similarity score from vector search
    - max_similarity(doc, selected_docs): Maximum similarity to already selected documents

This greedy algorithm iteratively selects documents that maximize the MMR score,
ensuring both relevance to the query and dissimilarity to previously selected results.
"""

import numpy as np
from loguru import logger
from numpy.typing import NDArray


def mmr_rerank(
    query_embedding: NDArray[np.float32],
    candidate_embeddings: NDArray[np.float32],
    candidate_scores: list[float],
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[int]:
    """Rerank candidates using MMR for diversity.

    Args:
        query_embedding: Query vector (1D array, shape: [dim])
        candidate_embeddings: Matrix of candidate vectors (2D array, shape: [n_candidates, dim])
        candidate_scores: Original relevance scores for each candidate (length: n_candidates)
        lambda_param: Balance between relevance and diversity
            - 1.0: Pure relevance (same as original ranking)
            - 0.7: Balanced (default, good for most cases)
            - 0.5: Equal weight for relevance and diversity
            - 0.0: Pure diversity (maximally different results)
        top_k: Number of results to return (must be <= len(candidate_scores))

    Returns:
        List of indices into candidates, ordered by MMR score (length: min(top_k, n_candidates))

    Raises:
        ValueError: If inputs have invalid shapes or dimensions don't match

    Example:
        >>> import numpy as np
        >>> query = np.random.randn(384).astype(np.float32)
        >>> candidates = np.random.randn(20, 384).astype(np.float32)
        >>> scores = [0.9, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80,
        ...           0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70]
        >>> indices = mmr_rerank(query, candidates, scores, lambda_param=0.7, top_k=10)
        >>> len(indices)
        10
    """
    # Input validation
    if query_embedding.ndim != 1:
        raise ValueError(
            f"query_embedding must be 1D, got shape {query_embedding.shape}"
        )

    if candidate_embeddings.ndim != 2:
        raise ValueError(
            f"candidate_embeddings must be 2D, got shape {candidate_embeddings.shape}"
        )

    n_candidates, dim = candidate_embeddings.shape

    if query_embedding.shape[0] != dim:
        raise ValueError(
            f"Dimension mismatch: query has {query_embedding.shape[0]}D, "
            f"candidates have {dim}D"
        )

    if len(candidate_scores) != n_candidates:
        raise ValueError(
            f"Scores length ({len(candidate_scores)}) must match candidates ({n_candidates})"
        )

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    # Edge cases: return early if possible
    if n_candidates == 0:
        logger.debug("MMR: No candidates to rerank")
        return []

    if n_candidates == 1:
        logger.debug("MMR: Single candidate, returning as-is")
        return [0]

    # Limit top_k to available candidates
    top_k = min(top_k, n_candidates)

    # Normalize embeddings for cosine similarity (dot product on normalized vectors)
    # This is more efficient than computing cosine similarity each time
    candidate_norms = candidate_embeddings / (
        np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10
    )

    # Pre-compute pairwise similarities between all candidates
    # Shape: [n_candidates, n_candidates]
    # similarity_matrix[i][j] = cosine similarity between candidate i and candidate j
    similarity_matrix = candidate_norms @ candidate_norms.T

    # Initialize selected indices and unselected mask
    selected_indices: list[int] = []
    unselected_mask = np.ones(n_candidates, dtype=bool)

    # Greedy selection: iteratively select document with highest MMR score
    for _ in range(top_k):
        if not np.any(unselected_mask):
            break

        # Calculate MMR score for all unselected candidates
        mmr_scores = np.full(n_candidates, -np.inf)

        for idx in range(n_candidates):
            if not unselected_mask[idx]:
                continue

            # Relevance component: original similarity score from vector search
            relevance = candidate_scores[idx]

            # Diversity component: max similarity to already selected documents
            if selected_indices:
                # Get similarities between this candidate and all selected documents
                similarities_to_selected = similarity_matrix[idx, selected_indices]
                max_similarity = np.max(similarities_to_selected)
            else:
                # No documents selected yet, diversity penalty is 0
                max_similarity = 0.0

            # MMR formula: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity
            mmr_scores[idx] = mmr_score

        # Select candidate with highest MMR score
        best_idx = int(np.argmax(mmr_scores))
        selected_indices.append(best_idx)
        unselected_mask[best_idx] = False

        logger.trace(
            f"MMR iteration {len(selected_indices)}/{top_k}: "
            f"selected idx={best_idx}, "
            f"relevance={candidate_scores[best_idx]:.3f}, "
            f"mmr_score={mmr_scores[best_idx]:.3f}"
        )

    logger.debug(
        f"MMR reranking complete: selected {len(selected_indices)} from {n_candidates} candidates "
        f"(lambda={lambda_param:.2f})"
    )

    return selected_indices


def mmr_rerank_with_metadata(
    query_embedding: NDArray[np.float32],
    candidates: list[dict],
    embedding_key: str = "embedding",
    score_key: str = "score",
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[dict]:
    """Rerank candidates with metadata using MMR.

    Convenience wrapper around mmr_rerank() that handles dictionaries with embeddings.

    Args:
        query_embedding: Query vector (1D array)
        candidates: List of candidate dictionaries containing embeddings and scores
        embedding_key: Key for embedding vector in candidate dict (default: "embedding")
        score_key: Key for relevance score in candidate dict (default: "score")
        lambda_param: Balance between relevance and diversity (default: 0.7)
        top_k: Number of results to return (default: 10)

    Returns:
        List of reranked candidate dictionaries (length: min(top_k, len(candidates)))

    Example:
        >>> candidates = [
        ...     {"embedding": np.random.randn(384), "score": 0.9, "content": "..."},
        ...     {"embedding": np.random.randn(384), "score": 0.8, "content": "..."},
        ... ]
        >>> query = np.random.randn(384).astype(np.float32)
        >>> reranked = mmr_rerank_with_metadata(query, candidates, top_k=5)
    """
    if not candidates:
        return []

    # Extract embeddings and scores
    embeddings = np.array([c[embedding_key] for c in candidates], dtype=np.float32)
    scores = [c[score_key] for c in candidates]

    # Rerank using MMR
    selected_indices = mmr_rerank(
        query_embedding=query_embedding,
        candidate_embeddings=embeddings,
        candidate_scores=scores,
        lambda_param=lambda_param,
        top_k=top_k,
    )

    # Return reordered candidates
    return [candidates[idx] for idx in selected_indices]
