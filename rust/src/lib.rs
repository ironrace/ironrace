// PyO3 0.22 map_err patterns trigger false positives with clippy 1.93+
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;

mod assembler;
mod json_fast;
mod pipeline;
mod tokenizer;
mod vector;

/// Returns the version of the ironrace package.
#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

/// IronRace Rust core module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<vector::VectorIndex>()?;
    m.add_function(wrap_pyfunction!(json_fast::parse_json, m)?)?;
    m.add_function(wrap_pyfunction!(json_fast::serialize_json, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer::truncate_to_budget, m)?)?;
    m.add_function(wrap_pyfunction!(assembler::assemble_prompt, m)?)?;
    m.add_class::<assembler::AssemblyResult>()?;
    m.add_function(wrap_pyfunction!(pipeline::execute_pipeline, m)?)?;
    Ok(())
}
