"""Compiler that analyzes decorated components and builds execution DAGs."""

__all__ = ["compile_context", "compile_pipeline", "compile_agents_dag"]

import json

from ironrace.decorators import _compiled_dags, get_registry
from ironrace.types import APIFetch, Feature, VectorSearch


def compile_context(context_cls: type) -> list[dict]:
    """Build DAG nodes for a context class's data dependencies."""
    nodes = []
    info = get_registry()["contexts"].get(context_cls.__name__)
    if not info:
        return nodes

    for field_name, descriptor in info["descriptors"].items():
        if isinstance(descriptor, VectorSearch):
            nodes.append(
                {
                    "id": f"ctx_{context_cls.__name__}_{field_name}",
                    "op": {
                        "type": "vector_search",
                        "vectors": [],  # placeholder — populated at runtime
                        "query": [],
                        "top_k": descriptor.top_k,
                    },
                    "depends_on": [],
                    "field": field_name,
                    "descriptor_type": "vector_search",
                }
            )
        elif isinstance(descriptor, APIFetch):
            nodes.append(
                {
                    "id": f"ctx_{context_cls.__name__}_{field_name}",
                    "op": {
                        "type": "passthrough",
                        "data": None,  # populated at runtime with API response
                    },
                    "depends_on": [],
                    "field": field_name,
                    "descriptor_type": "api_fetch",
                }
            )
        elif isinstance(descriptor, Feature):
            # Features depend on other context fields
            dep_ids = [
                f"ctx_{context_cls.__name__}_{f}"
                for f in info["descriptors"]
                if f != field_name
                and not isinstance(info["descriptors"][f], Feature)
            ]
            nodes.append(
                {
                    "id": f"ctx_{context_cls.__name__}_{field_name}",
                    "op": {"type": "passthrough", "data": None},
                    "depends_on": dep_ids,
                    "field": field_name,
                    "descriptor_type": "feature",
                }
            )

    return nodes


def compile_pipeline(pipeline_name: str) -> dict:
    """Compile a pipeline into a DAG definition.

    Analyzes the registry to build a dependency graph from context
    descriptors and agent relationships.

    Returns:
        A dict representing the DAG suitable for the Rust executor.
    """
    if pipeline_name in _compiled_dags:
        return _compiled_dags[pipeline_name]

    registry = get_registry()
    pipeline_info = registry["pipelines"].get(pipeline_name)
    if not pipeline_info:
        raise ValueError(f"Pipeline '{pipeline_name}' not found in registry")

    all_nodes = []
    agent_nodes = []

    # Gather all agents and their context dependencies
    for agent_name, agent_info in registry["agents"].items():
        context_cls = agent_info.get("context")

        # Add context prep nodes
        if context_cls:
            ctx_nodes = compile_context(context_cls)
            all_nodes.extend(ctx_nodes)
            context_deps = [n["id"] for n in ctx_nodes]
        else:
            context_deps = []

        # Add assembly node for the agent's prompt
        assembly_node = {
            "id": f"agent_{agent_name}_assemble",
            "op": {
                "type": "assemble",
                "template": "",  # populated at runtime from agent function
                "values": {},
                "budgets": {},
            },
            "depends_on": context_deps,
            "agent": agent_name,
        }
        all_nodes.append(assembly_node)
        agent_nodes.append(assembly_node)

    dag = {
        "nodes": all_nodes,
        "agents": [n["id"] for n in agent_nodes],
        "pipeline": pipeline_name,
    }

    _compiled_dags[pipeline_name] = dag
    return dag


def compile_agents_dag(
    agents: list[dict], shared_data: dict | None = None
) -> str:
    """Build a DAG JSON string for a set of agent context preparation steps.

    This is the main entry point for creating execution plans that cross
    the Python-Rust bridge once.

    Args:
        agents: List of agent specs with template, values, budgets.
        shared_data: Optional shared data (e.g., vector search results)
                     available to all agents.

    Returns:
        JSON string suitable for execute_pipeline().
    """
    nodes = []

    for i, agent_spec in enumerate(agents):
        agent_id = agent_spec.get("id", f"agent_{i}")
        depends = agent_spec.get("depends_on", [])

        if "template" in agent_spec:
            nodes.append(
                {
                    "id": agent_id,
                    "op": {
                        "type": "assemble",
                        "template": agent_spec["template"],
                        "values": agent_spec.get("values", {}),
                        "budgets": agent_spec.get("budgets", {}),
                        "model": agent_spec.get("model", "approximate"),
                    },
                    "depends_on": depends,
                }
            )
        elif "op" in agent_spec:
            nodes.append(
                {
                    "id": agent_id,
                    "op": agent_spec["op"],
                    "depends_on": depends,
                }
            )

    dag = {"nodes": nodes}
    return json.dumps(dag)
