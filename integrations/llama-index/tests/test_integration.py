"""Integration tests with LlamaIndex core (VectorStoreIndex, retriever)."""

import math
import os
import random

import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import QueryBundle, TextNode

from llama_index.vector_stores.ironrace import IronRaceVectorStore

# Tell LlamaIndex to use a mock embed model (no OpenAI key needed)
os.environ["IS_TESTING"] = "true"


def _make_nodes(n=10, dim=8):
    """Create TextNodes with random embeddings.

    Uses dim=8 to match LlamaIndex's MockEmbedding(embed_dim=8).
    """
    nodes = []
    for i in range(n):
        random.seed(i)
        vec = [random.gauss(0, 1) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec))
        nodes.append(
            TextNode(
                text=f"Document {i}: This is test content about topic {i}.",
                id_=f"doc_{i}",
                embedding=[x / norm for x in vec],
                metadata={"index": i, "topic": f"topic_{i % 5}"},
            )
        )
    return nodes


class TestVectorStoreIndex:
    def test_construction(self):
        """VectorStoreIndex should accept IronRaceVectorStore via StorageContext."""
        vector_store = IronRaceVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = _make_nodes(10)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )
        assert index is not None

    def test_retriever_retrieve(self):
        """Retriever should return NodeWithScore objects."""
        vector_store = IronRaceVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        nodes = _make_nodes(20)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )

        retriever = index.as_retriever(similarity_top_k=5)

        # Use a known node's embedding as the query
        query_bundle = QueryBundle(
            query_str="test",
            embedding=nodes[3].embedding,
        )
        results = retriever.retrieve(query_bundle)

        assert len(results) == 5
        assert hasattr(results[0], "score")
        assert hasattr(results[0], "node")
        # The queried node should be in the top results
        result_ids = [r.node.node_id for r in results]
        assert "doc_3" in result_ids

    def test_storage_context_wiring(self):
        """StorageContext should properly wire the vector store."""
        vector_store = IronRaceVectorStore(ef_construction=20)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        assert storage_context.vector_store is vector_store
        assert storage_context.vector_store.ef_construction == 20
