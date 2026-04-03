use pyo3::prelude::*;
use std::collections::HashMap;

use crate::tokenizer;

/// Result of prompt assembly, including the final prompt and metadata.
#[pyclass]
#[derive(Clone)]
pub struct AssemblyResult {
    #[pyo3(get)]
    pub prompt: String,
    #[pyo3(get)]
    pub total_tokens: usize,
    #[pyo3(get)]
    pub sections_truncated: Vec<String>,
    #[pyo3(get)]
    pub token_breakdown: HashMap<String, usize>,
}

#[pymethods]
impl AssemblyResult {
    fn __repr__(&self) -> String {
        format!(
            "AssemblyResult(total_tokens={}, sections_truncated={:?})",
            self.total_tokens, self.sections_truncated
        )
    }
}

/// Assemble a prompt from a template and values, enforcing per-section token budgets.
///
/// Template uses `{variable}` placeholders. Each variable's value can have an optional
/// token budget — values exceeding their budget are truncated at sentence boundaries.
///
/// Args:
///     template: Template string with `{variable}` placeholders.
///     values: Mapping of variable names to their string values.
///     budgets: Mapping of variable names to their max token budgets.
///              Variables without a budget entry are inserted as-is.
///     model: Tokenizer model for counting tokens (default: "approximate").
///
/// Returns:
///     AssemblyResult with the assembled prompt and metadata.
#[pyfunction]
#[pyo3(signature = (template, values, budgets, model="approximate"))]
pub fn assemble_prompt(
    template: &str,
    values: HashMap<String, String>,
    budgets: HashMap<String, usize>,
    model: &str,
) -> AssemblyResult {
    let mut result = template.to_string();
    let mut sections_truncated = Vec::new();
    let mut token_breakdown = HashMap::new();

    // Process each value: truncate if budget exists, then interpolate
    for (key, value) in &values {
        let placeholder = format!("{{{}}}", key);
        if !result.contains(&placeholder) {
            continue;
        }

        let final_value = if let Some(&budget) = budgets.get(key) {
            let truncated = tokenizer::truncate_to_budget(value, budget, model);
            if truncated.len() < value.len() {
                sections_truncated.push(key.clone());
            }
            truncated
        } else {
            value.clone()
        };

        let tokens = tokenizer::count_tokens(&final_value, model);
        token_breakdown.insert(key.clone(), tokens);
        result = result.replace(&placeholder, &final_value);
    }

    // Count total tokens of assembled prompt
    let total_tokens = tokenizer::count_tokens(&result, model);

    AssemblyResult {
        prompt: result,
        total_tokens,
        sections_truncated,
        token_breakdown,
    }
}
