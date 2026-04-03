use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::runtime::Runtime;

use crate::{assembler, tokenizer, vector};

static RUNTIME: once_cell::sync::Lazy<Runtime> = once_cell::sync::Lazy::new(|| {
    Runtime::new().expect("Failed to create Tokio runtime")
});

/// Supported operations that can be executed as DAG nodes.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type")]
pub enum Operation {
    #[serde(rename = "vector_search")]
    VectorSearch {
        vectors: Vec<Vec<f32>>,
        query: Vec<f32>,
        top_k: usize,
        #[serde(default = "default_ef")]
        ef_construction: usize,
    },
    #[serde(rename = "json_parse")]
    JsonParse { data: String },
    #[serde(rename = "json_serialize")]
    JsonSerialize { data: serde_json::Value },
    #[serde(rename = "count_tokens")]
    CountTokens {
        text: String,
        #[serde(default = "default_model")]
        model: String,
    },
    #[serde(rename = "truncate")]
    Truncate {
        text: String,
        max_tokens: usize,
        #[serde(default = "default_model")]
        model: String,
    },
    #[serde(rename = "assemble")]
    Assemble {
        template: String,
        values: HashMap<String, String>,
        #[serde(default)]
        budgets: HashMap<String, usize>,
        #[serde(default = "default_model")]
        model: String,
    },
    #[serde(rename = "passthrough")]
    Passthrough { data: serde_json::Value },
}

fn default_ef() -> usize {
    200
}
fn default_model() -> String {
    "approximate".to_string()
}

/// A node in the execution DAG.
#[derive(Debug, Deserialize, Clone)]
pub struct DAGNode {
    pub id: String,
    pub op: Operation,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

/// The full DAG definition passed from Python.
#[derive(Debug, Deserialize)]
pub struct DAGDefinition {
    pub nodes: Vec<DAGNode>,
}

/// Execute a single operation, returning the result as JSON value.
fn execute_operation(op: &Operation) -> serde_json::Value {
    match op {
        Operation::VectorSearch {
            vectors,
            query,
            top_k,
            ef_construction,
        } => {
            let idx = vector::VectorIndex::new(vectors.clone(), *ef_construction);
            let results = idx.search(query.clone(), *top_k);
            serde_json::json!(results
                .iter()
                .map(|(i, s)| serde_json::json!({"index": i, "score": s}))
                .collect::<Vec<_>>())
        }
        Operation::JsonParse { data } => {
            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(v) => v,
                Err(e) => serde_json::json!({"error": e.to_string()}),
            }
        }
        Operation::JsonSerialize { data } => {
            serde_json::json!({"bytes": serde_json::to_string(data).unwrap_or_default()})
        }
        Operation::CountTokens { text, model } => {
            let count = tokenizer::count_tokens(text, model);
            serde_json::json!({"count": count})
        }
        Operation::Truncate {
            text,
            max_tokens,
            model,
        } => {
            let truncated = tokenizer::truncate_to_budget(text, *max_tokens, model);
            serde_json::json!({"text": truncated})
        }
        Operation::Assemble {
            template,
            values,
            budgets,
            model,
        } => {
            let result =
                assembler::assemble_prompt(template, values.clone(), budgets.clone(), model);
            serde_json::json!({
                "prompt": result.prompt,
                "total_tokens": result.total_tokens,
                "sections_truncated": result.sections_truncated,
                "token_breakdown": result.token_breakdown,
            })
        }
        Operation::Passthrough { data } => data.clone(),
    }
}

/// Topological sort of DAG nodes. Returns groups of nodes that can run in parallel.
fn topological_groups(nodes: &[DAGNode]) -> Vec<Vec<usize>> {
    let n = nodes.len();
    let id_to_idx: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.id.as_str(), i))
        .collect();

    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];

    for (i, node) in nodes.iter().enumerate() {
        for dep in &node.depends_on {
            if let Some(&dep_idx) = id_to_idx.get(dep.as_str()) {
                in_degree[i] += 1;
                dependents[dep_idx].push(i);
            }
        }
    }

    let mut groups = Vec::new();
    let mut completed = vec![false; n];

    loop {
        let ready: Vec<usize> = (0..n)
            .filter(|&i| !completed[i] && in_degree[i] == 0)
            .collect();

        if ready.is_empty() {
            break;
        }

        for &i in &ready {
            completed[i] = true;
            for &dep in &dependents[i] {
                in_degree[dep] -= 1;
            }
        }

        groups.push(ready);
    }

    groups
}

/// Execute a DAG of context preparation operations.
///
/// Takes a JSON string defining the DAG and returns a JSON string with results.
/// All operations run in Rust — this crosses the Python-Rust bridge only once.
///
/// Args:
///     dag_json: JSON string containing a DAGDefinition with nodes and dependencies.
///
/// Returns:
///     JSON string mapping node IDs to their result values.
#[pyfunction]
pub fn execute_pipeline(dag_json: &str) -> PyResult<String> {
    let dag: DAGDefinition = serde_json::from_str(dag_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid DAG JSON: {e}"))
    })?;

    let groups = topological_groups(&dag.nodes);
    let mut results: HashMap<String, serde_json::Value> = HashMap::new();

    for group in groups {
        if group.len() == 1 {
            // Single node — execute directly
            let node = &dag.nodes[group[0]];
            let result = execute_operation(&node.op);
            results.insert(node.id.clone(), result);
        } else {
            // Multiple independent nodes — execute in parallel via Tokio
            let nodes_for_group: Vec<DAGNode> =
                group.iter().map(|&i| dag.nodes[i].clone()).collect();

            let group_results: Vec<(String, serde_json::Value)> =
                RUNTIME.block_on(async {
                    let mut handles = Vec::new();
                    for node in nodes_for_group {
                        handles.push(tokio::spawn(async move {
                            let result = execute_operation(&node.op);
                            (node.id, result)
                        }));
                    }
                    let mut results = Vec::new();
                    for handle in handles {
                        results.push(handle.await.unwrap());
                    }
                    results
                });

            for (id, result) in group_results {
                results.insert(id, result);
            }
        }
    }

    serde_json::to_string(&results).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to serialize results: {e}"))
    })
}
