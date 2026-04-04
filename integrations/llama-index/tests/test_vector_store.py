"""Unit tests for IronRaceVectorStore."""

import math
import random

import pytest
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.ironrace import IronRaceVectorStore


@pytest.fixture
def tmp_persist_path(tmp_path):
    return str(tmp_path / "test_store.json")


def _make_node(text="hello", dim=64, node_id=None, metadata=None, ref_doc_id=None, seed=None):
    """Create a TextNode with a random unit-vector embedding."""
    if seed is not None:
        random.seed(seed)
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    embedding = [x / norm for x in vec]

    node = TextNode(
        text=text,
        id_=node_id or f"node_{random.randint(0, 999999)}",
        embedding=embedding,
        metadata=metadata or {},
    )
    if ref_doc_id:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
            node_id=ref_doc_id
        )
    return node


class TestAdd:
    def test_add_nodes(self):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=i) for i in range(10)]
        ids = store.add(nodes)
        assert len(ids) == 10

    def test_add_returns_node_ids(self):
        store = IronRaceVectorStore()
        node = _make_node(node_id="test_123")
        ids = store.add([node])
        assert ids == ["test_123"]

    def test_add_no_embedding_raises(self):
        store = IronRaceVectorStore()
        node = TextNode(text="no embedding", id_="bad")
        with pytest.raises(ValueError, match="embedding"):
            store.add([node])

    def test_embedding_dimension_mismatch(self):
        store = IronRaceVectorStore()
        store.add([_make_node(dim=64, seed=0)])
        with pytest.raises(ValueError, match="dimension mismatch"):
            store.add([_make_node(dim=128, seed=1)])


class TestQuery:
    def test_query_self_match(self):
        store = IronRaceVectorStore()
        node = _make_node(seed=42, node_id="target")
        store.add([node])

        query = VectorStoreQuery(
            query_embedding=node.embedding,
            similarity_top_k=1,
        )
        result = store.query(query)

        assert len(result.nodes) == 1
        assert result.ids[0] == "target"
        assert result.similarities[0] > 0.99

    def test_query_top_k(self):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=i) for i in range(50)]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=5,
        )
        result = store.query(query)
        assert len(result.nodes) == 5

    def test_query_sorted_by_score(self):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=i) for i in range(30)]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
        )
        result = store.query(query)
        scores = result.similarities
        assert scores == sorted(scores, reverse=True)

    def test_empty_store_query(self):
        store = IronRaceVectorStore()
        query = VectorStoreQuery(
            query_embedding=[0.0] * 64,
            similarity_top_k=5,
        )
        result = store.query(query)
        assert result.nodes == []
        assert result.similarities == []
        assert result.ids == []

    def test_top_k_greater_than_store_size(self):
        """HNSW is approximate — may not return every vector at tiny scale."""
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=i) for i in range(20)]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=100,
        )
        result = store.query(query)
        # HNSW is approximate — at 20 vectors, graph isolation can lose 1-3 vectors
        assert len(result.nodes) >= 17

    def test_high_dimensional(self):
        """Test with 1536-d vectors (OpenAI embedding size)."""
        store = IronRaceVectorStore()
        nodes = [_make_node(dim=1536, seed=i) for i in range(20)]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[5].embedding,
            similarity_top_k=3,
        )
        result = store.query(query)
        assert len(result.nodes) == 3
        assert result.ids[0] == nodes[5].node_id


class TestDelete:
    def test_delete_by_ref_doc_id(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="n0", ref_doc_id="doc_A"),
            _make_node(seed=1, node_id="n1", ref_doc_id="doc_A"),
            _make_node(seed=2, node_id="n2", ref_doc_id="doc_B"),
        ]
        store.add(nodes)

        store.delete("doc_A")

        # Only doc_B node should remain
        query = VectorStoreQuery(
            query_embedding=nodes[2].embedding,
            similarity_top_k=10,
        )
        result = store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "n2"

    def test_delete_nonexistent_doc(self):
        store = IronRaceVectorStore()
        store.add([_make_node(seed=0)])
        store.delete("nonexistent")  # should not raise


class TestIncrementalAdd:
    def test_add_then_query_then_add_again(self):
        store = IronRaceVectorStore()
        batch1 = [_make_node(seed=i) for i in range(10)]
        store.add(batch1)

        # Query works after first batch
        result = store.query(VectorStoreQuery(
            query_embedding=batch1[0].embedding, similarity_top_k=5
        ))
        assert len(result.nodes) == 5

        # Add second batch
        batch2 = [_make_node(seed=i + 100) for i in range(10)]
        store.add(batch2)

        # Query sees all 20 nodes
        result = store.query(VectorStoreQuery(
            query_embedding=batch1[0].embedding, similarity_top_k=20
        ))
        assert len(result.nodes) == 20


class TestMetadataFilters:
    def test_filter_eq(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="a", metadata={"category": "tech"}),
            _make_node(seed=1, node_id="b", metadata={"category": "finance"}),
            _make_node(seed=2, node_id="c", metadata={"category": "tech"}),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="category", value="tech", operator=FilterOperator.EQ)]
            ),
        )
        result = store.query(query)
        assert all(n.metadata["category"] == "tech" for n in result.nodes)

    def test_filter_gt(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="a", metadata={"score": 10}),
            _make_node(seed=1, node_id="b", metadata={"score": 50}),
            _make_node(seed=2, node_id="c", metadata={"score": 90}),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="score", value=40, operator=FilterOperator.GT)]
            ),
        )
        result = store.query(query)
        assert all(n.metadata["score"] > 40 for n in result.nodes)

    def test_filter_and(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="a", metadata={"cat": "tech", "score": 80}),
            _make_node(seed=1, node_id="b", metadata={"cat": "tech", "score": 20}),
            _make_node(seed=2, node_id="c", metadata={"cat": "finance", "score": 90}),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="cat", value="tech", operator=FilterOperator.EQ),
                    MetadataFilter(key="score", value=50, operator=FilterOperator.GT),
                ],
                condition=FilterCondition.AND,
            ),
        )
        result = store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "a"

    def test_filter_or(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="a", metadata={"cat": "tech"}),
            _make_node(seed=1, node_id="b", metadata={"cat": "finance"}),
            _make_node(seed=2, node_id="c", metadata={"cat": "health"}),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(key="cat", value="tech", operator=FilterOperator.EQ),
                    MetadataFilter(key="cat", value="health", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )
        result = store.query(query)
        categories = {n.metadata["cat"] for n in result.nodes}
        assert categories <= {"tech", "health"}


class TestDeleteNodes:
    def test_delete_nodes_by_id(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="n0"),
            _make_node(seed=1, node_id="n1"),
            _make_node(seed=2, node_id="n2"),
        ]
        store.add(nodes)

        store.delete_nodes(["n0", "n2"])

        query = VectorStoreQuery(
            query_embedding=nodes[1].embedding, similarity_top_k=10
        )
        result = store.query(query)
        assert len(result.nodes) == 1
        assert result.ids[0] == "n1"

    def test_delete_nodes_nonexistent(self):
        store = IronRaceVectorStore()
        store.add([_make_node(seed=0, node_id="n0")])
        store.delete_nodes(["nonexistent"])  # should not raise

        query = VectorStoreQuery(
            query_embedding=[0.0] * 64, similarity_top_k=10
        )
        result = store.query(query)
        assert len(result.nodes) == 1

    def test_delete_nodes_cleans_ref_doc_mapping(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="n0", ref_doc_id="doc_A"),
            _make_node(seed=1, node_id="n1", ref_doc_id="doc_A"),
        ]
        store.add(nodes)

        store.delete_nodes(["n0", "n1"])

        # ref_doc_id mapping should be cleaned up
        assert "doc_A" not in store._ref_doc_id_to_node_ids


class TestClear:
    def test_clear_removes_all(self):
        store = IronRaceVectorStore()
        store.add([_make_node(seed=i) for i in range(10)])

        store.clear()

        query = VectorStoreQuery(
            query_embedding=[0.0] * 64, similarity_top_k=10
        )
        result = store.query(query)
        assert result.nodes == []

    def test_clear_resets_embedding_dim(self):
        store = IronRaceVectorStore()
        store.add([_make_node(dim=64, seed=0)])
        store.clear()
        # Should accept different dimension after clear
        store.add([_make_node(dim=128, seed=1)])
        assert store._embedding_dim == 128


class TestEdgeCases:
    def test_all_filtered_out(self):
        store = IronRaceVectorStore()
        nodes = [
            _make_node(seed=0, node_id="a", metadata={"status": "inactive"}),
            _make_node(seed=1, node_id="b", metadata={"status": "inactive"}),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=10,
            filters=MetadataFilters(
                filters=[MetadataFilter(key="status", value="active", operator=FilterOperator.EQ)]
            ),
        )
        result = store.query(query)
        assert result.nodes == []

    def test_query_no_embedding_raises(self):
        store = IronRaceVectorStore()
        store.add([_make_node(seed=0)])

        with pytest.raises(ValueError, match="query_embedding"):
            store.query(VectorStoreQuery(query_embedding=None, similarity_top_k=5))

    def test_corrupted_persist_file(self, tmp_path):
        persist_path = str(tmp_path / "bad.json")
        with open(persist_path, "w") as f:
            f.write("{invalid json")

        with pytest.raises(Exception):
            IronRaceVectorStore.from_persist_path(persist_path)


class TestPersistence:
    def test_persist_and_load(self, tmp_persist_path):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=i, node_id=f"n{i}") for i in range(20)]
        store.add(nodes)

        # Persist
        store.persist(tmp_persist_path)

        # Load into new store
        loaded = IronRaceVectorStore.from_persist_path(tmp_persist_path)

        # Query should work and return results
        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding,
            similarity_top_k=5,
        )
        result = loaded.query(query)
        assert len(result.nodes) == 5
        assert result.ids[0] == "n0"
        assert result.similarities[0] > 0.99

    def test_persist_preserves_metadata(self, tmp_persist_path):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=0, node_id="n0", metadata={"key": "value", "num": 42})]
        store.add(nodes)
        store.persist(tmp_persist_path)

        loaded = IronRaceVectorStore.from_persist_path(tmp_persist_path)
        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding, similarity_top_k=1
        )
        result = loaded.query(query)
        assert result.nodes[0].metadata["key"] == "value"
        assert result.nodes[0].metadata["num"] == 42

    def test_persist_preserves_text(self, tmp_persist_path):
        store = IronRaceVectorStore()
        nodes = [_make_node(seed=0, node_id="n0", text="Hello world content")]
        store.add(nodes)
        store.persist(tmp_persist_path)

        loaded = IronRaceVectorStore.from_persist_path(tmp_persist_path)
        query = VectorStoreQuery(
            query_embedding=nodes[0].embedding, similarity_top_k=1
        )
        result = loaded.query(query)
        assert "Hello world content" in result.nodes[0].get_content()

    def test_persist_preserves_ef_construction(self, tmp_persist_path):
        store = IronRaceVectorStore(ef_construction=80)
        store.add([_make_node(seed=0)])
        store.persist(tmp_persist_path)

        loaded = IronRaceVectorStore.from_persist_path(tmp_persist_path)
        assert loaded.ef_construction == 80

    def test_load_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No IronRaceVectorStore"):
            IronRaceVectorStore.from_persist_path("/nonexistent/path.json")

    def test_persist_empty_store(self, tmp_persist_path):
        store = IronRaceVectorStore()
        store.persist(tmp_persist_path)

        loaded = IronRaceVectorStore.from_persist_path(tmp_persist_path)
        result = loaded.query(VectorStoreQuery(
            query_embedding=[0.0] * 64, similarity_top_k=5
        ))
        assert result.nodes == []
