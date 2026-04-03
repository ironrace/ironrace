"""IronRace vector store — Rust HNSW backend for LlamaIndex."""

import logging
from collections import defaultdict
from typing import Any, List, Optional, Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from pydantic import Field, PrivateAttr

from ironrace import VectorIndex

logger = logging.getLogger(__name__)


class IronRaceVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by IronRace's Rust HNSW index.

    Drop-in replacement for SimpleVectorStore with significantly faster
    similarity search via Rust-native HNSW (instant-distance crate).

    The index is built lazily on first query and rebuilt automatically
    when nodes are added or deleted.

    Example:
        from llama_index.vector_stores.ironrace import IronRaceVectorStore
        from llama_index.core import VectorStoreIndex, StorageContext

        vector_store = IronRaceVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=documents, storage_context=storage_context)
        results = index.as_retriever(similarity_top_k=10).retrieve("query")
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    ef_construction: int = Field(
        default=40,
        description="HNSW build quality. Higher = better recall, slower rebuilds.",
    )
    similarity_top_k_multiplier: float = Field(
        default=3.0,
        description="Over-fetch multiplier when metadata filters are applied.",
    )

    _index: Optional[VectorIndex] = PrivateAttr(default=None)
    _embeddings: list = PrivateAttr(default_factory=list)
    _node_ids: list = PrivateAttr(default_factory=list)
    _nodes: dict = PrivateAttr(default_factory=dict)
    _ref_doc_id_to_node_ids: dict = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )
    _dirty: bool = PrivateAttr(default=True)
    _embedding_dim: Optional[int] = PrivateAttr(default=None)

    @property
    def client(self) -> Any:
        """Return the Rust VectorIndex (rebuilding if needed)."""
        self._ensure_index()
        return self._index

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Add nodes with embeddings to the store.

        Args:
            nodes: Sequence of nodes. Each must have a non-None embedding.

        Returns:
            List of added node IDs.
        """
        added_ids = []

        for node in nodes:
            embedding = node.get_embedding()
            if embedding is None:
                raise ValueError(
                    f"Node {node.node_id} has no embedding. "
                    "Ensure embeddings are computed before adding to IronRaceVectorStore."
                )

            # Validate embedding dimension consistency
            if self._embedding_dim is None:
                self._embedding_dim = len(embedding)
            elif len(embedding) != self._embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                    f"got {len(embedding)} for node {node.node_id}"
                )

            self._nodes[node.node_id] = node
            self._embeddings.append(embedding)
            self._node_ids.append(node.node_id)

            # Track ref_doc_id for delete()
            ref_doc_id = node.ref_doc_id
            if ref_doc_id:
                self._ref_doc_id_to_node_ids[ref_doc_id].append(node.node_id)

            added_ids.append(node.node_id)

        self._dirty = True
        return added_ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete all nodes associated with a document reference ID."""
        node_ids_to_delete = self._ref_doc_id_to_node_ids.pop(ref_doc_id, [])
        if not node_ids_to_delete:
            return

        delete_set = set(node_ids_to_delete)
        for nid in node_ids_to_delete:
            self._nodes.pop(nid, None)

        # Rebuild embedding and ID lists without deleted nodes
        new_embeddings = []
        new_node_ids = []
        for nid, emb in zip(self._node_ids, self._embeddings):
            if nid not in delete_set:
                new_embeddings.append(emb)
                new_node_ids.append(nid)

        self._embeddings = new_embeddings
        self._node_ids = new_node_ids
        self._dirty = True

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Query the vector store for similar nodes.

        Uses Rust HNSW for fast approximate nearest neighbor search.
        Metadata filters are applied post-search with over-fetching.
        """
        if not self._embeddings:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if query.query_embedding is None:
            raise ValueError("query_embedding is required for IronRaceVectorStore")

        self._ensure_index()

        top_k = query.similarity_top_k
        has_filters = query.filters is not None and len(query.filters.filters) > 0

        # Over-fetch if we need to filter
        fetch_k = (
            int(top_k * self.similarity_top_k_multiplier)
            if has_filters
            else top_k
        )
        fetch_k = min(fetch_k, len(self._embeddings))

        # Rust HNSW search
        raw_results = self._index.search(query.query_embedding, fetch_k)

        # Convert results
        result_nodes = []
        result_similarities = []
        result_ids = []

        for idx, score in raw_results:
            if idx >= len(self._node_ids):
                continue

            node_id = self._node_ids[idx]
            node = self._nodes.get(node_id)
            if node is None:
                continue

            # Apply metadata filters
            if has_filters and not self._passes_filters(node, query.filters):
                continue

            result_nodes.append(node)
            result_similarities.append(score)
            result_ids.append(node_id)

            if len(result_nodes) >= top_k:
                break

        return VectorStoreQueryResult(
            nodes=result_nodes,
            similarities=result_similarities,
            ids=result_ids,
        )

    def _ensure_index(self) -> None:
        """Rebuild the Rust HNSW index if dirty."""
        if self._dirty and self._embeddings:
            n = len(self._embeddings)
            if n > 5000:
                logger.warning(
                    f"Rebuilding HNSW index with {n} vectors "
                    f"(ef_construction={self.ef_construction}). "
                    "This may take a moment. Consider batching adds."
                )
            self._index = VectorIndex(
                self._embeddings, self.ef_construction
            )
            self._dirty = False

    @staticmethod
    def _passes_filters(
        node: BaseNode, filters: MetadataFilters
    ) -> bool:
        """Check if a node passes the metadata filters."""
        results = [
            IronRaceVectorStore._evaluate_filter(node, f)
            for f in filters.filters
        ]

        if filters.condition == FilterCondition.OR:
            return any(results)
        else:  # AND (default)
            return all(results)

    @staticmethod
    def _evaluate_filter(node: BaseNode, f: MetadataFilter) -> bool:
        """Evaluate a single metadata filter against a node."""
        metadata = node.metadata or {}
        value = metadata.get(f.key)

        if value is None:
            return f.operator == FilterOperator.IS_EMPTY

        op = f.operator
        filter_val = f.value

        if op == FilterOperator.EQ:
            return value == filter_val
        elif op == FilterOperator.NE:
            return value != filter_val
        elif op == FilterOperator.GT:
            return value > filter_val
        elif op == FilterOperator.LT:
            return value < filter_val
        elif op == FilterOperator.GTE:
            return value >= filter_val
        elif op == FilterOperator.LTE:
            return value <= filter_val
        elif op == FilterOperator.IN:
            return value in (filter_val if isinstance(filter_val, list) else [filter_val])
        elif op == FilterOperator.NIN:
            return value not in (filter_val if isinstance(filter_val, list) else [filter_val])
        elif op == FilterOperator.CONTAINS:
            return filter_val in value if isinstance(value, (str, list)) else False
        elif op == FilterOperator.TEXT_MATCH:
            return str(filter_val).lower() in str(value).lower()
        elif op == FilterOperator.IS_EMPTY:
            return not value
        else:
            return True
