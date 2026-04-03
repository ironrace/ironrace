"""Tests for ironrace Python SDK (decorators, compiler, types)."""

import pytest

from ironrace import (
    APIFetch,
    Document,
    Feature,
    TokenBudget,
    VectorSearch,
    agent,
    compile_agents_dag,
    context,
    get_registry,
    pipeline,
)


# Reset registry between test modules
@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the registry around each test."""
    reg = get_registry()
    old = {k: dict(v) for k, v in reg.items()}
    yield
    for k in reg:
        reg[k].clear()
        reg[k].update(old[k])


class TestTypes:
    def test_token_budget(self):
        budget = TokenBudget(system=200, context=2000, user=400)
        assert budget.total == 2600

    def test_token_budget_defaults(self):
        budget = TokenBudget()
        assert budget.total == 4000

    def test_document(self):
        doc = Document(id="1", content="hello", metadata={"key": "val"}, score=0.9)
        assert doc.id == "1"
        assert doc.score == 0.9

    def test_vector_search_descriptor(self):
        vs = VectorSearch(collection="kb", top_k=20)
        assert vs.collection == "kb"
        assert vs.top_k == 20
        assert "VectorSearch" in repr(vs)

    def test_api_fetch_descriptor(self):
        af = APIFetch(url="https://api.example.com", params={"q": "test"})
        assert af.url == "https://api.example.com"
        assert "APIFetch" in repr(af)

    def test_feature_descriptor(self):
        f = Feature(lambda x: x + 1)
        assert callable(f.func)


class TestContextDecorator:
    def test_registers_context(self):
        @context
        class MyCtx:
            query: str
            docs: list = VectorSearch(collection="test", top_k=5)

        reg = get_registry()
        assert "MyCtx" in reg["contexts"]
        info = reg["contexts"]["MyCtx"]
        assert "docs" in info["descriptors"]
        assert isinstance(info["descriptors"]["docs"], VectorSearch)

    def test_context_with_multiple_descriptors(self):
        @context
        class MultiCtx:
            query: str
            docs: list = VectorSearch(collection="kb")
            api_data: dict = APIFetch(url="https://api.test.com")
            avg_score: float = Feature(lambda self: 0.5)

        info = get_registry()["contexts"]["MultiCtx"]
        assert len(info["descriptors"]) == 3

    def test_context_preserves_class(self):
        @context
        class PreservedCtx:
            name: str

        assert PreservedCtx.__name__ == "PreservedCtx"
        instance = PreservedCtx()
        instance.name = "test"
        assert instance.name == "test"


class TestAgentDecorator:
    def test_registers_agent(self):
        @agent(model="claude-sonnet-4-20250514", max_tokens=1000)
        def my_agent(query=""):
            return f"Analyze: {query}"

        reg = get_registry()
        assert "my_agent" in reg["agents"]
        assert reg["agents"]["my_agent"]["model"] == "claude-sonnet-4-20250514"
        assert reg["agents"]["my_agent"]["max_tokens"] == 1000

    def test_agent_callable(self):
        @agent()
        def simple_agent(query=""):
            return f"Q: {query}"

        result = simple_agent(query="test")
        assert result["prompt"] == "Q: test"
        assert result["agent"] == "simple_agent"

    def test_agent_with_context(self):
        @context
        class AgentCtx:
            idea: str

        @agent(context=AgentCtx)
        def ctx_agent(ctx, idea=""):
            return f"Idea: {idea}"

        reg = get_registry()
        assert reg["agents"]["ctx_agent"]["context"] == AgentCtx

    def test_agent_metadata(self):
        @agent(model="gpt-4", max_tokens=500)
        def meta_agent():
            return "prompt"

        assert meta_agent._af_type == "agent"
        assert meta_agent._af_model == "gpt-4"
        assert meta_agent._af_max_tokens == 500


class TestPipelineDecorator:
    def test_registers_pipeline(self):
        @pipeline(concurrency=50)
        def my_pipeline(idea=""):
            return {"idea": idea}

        reg = get_registry()
        assert "my_pipeline" in reg["pipelines"]
        assert reg["pipelines"]["my_pipeline"]["concurrency"] == 50

    def test_pipeline_callable(self):
        @pipeline()
        def simple_pipe(x=1):
            return x * 2

        result = simple_pipe(x=5)
        assert result == 10


class TestCompiler:
    def test_compile_agents_dag(self):
        agents = [
            {
                "id": "analyst",
                "template": "You are {role}. Analyze: {query}",
                "values": {"role": "security analyst", "query": "test idea"},
                "budgets": {"query": 100},
            },
            {
                "id": "market",
                "template": "Market data: {data}",
                "values": {"data": "market info"},
                "budgets": {},
            },
        ]

        import json

        dag_json = compile_agents_dag(agents)
        dag = json.loads(dag_json)

        assert len(dag["nodes"]) == 2
        assert dag["nodes"][0]["id"] == "analyst"
        assert dag["nodes"][1]["id"] == "market"

    def test_compiled_dag_executable(self):
        """Compiled DAG should be directly executable by the Rust pipeline."""
        from ironrace._core import execute_pipeline

        import json

        agents = [
            {
                "id": "agent1",
                "template": "Hello {name}",
                "values": {"name": "World"},
                "budgets": {},
            }
        ]

        dag_json = compile_agents_dag(agents)
        result = json.loads(execute_pipeline(dag_json))
        assert result["agent1"]["prompt"] == "Hello World"
