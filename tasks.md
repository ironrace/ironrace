# Tasks

Tasks are processed top-to-bottom by the Wiggum loop.
Format: `- [ ] Task name <!-- files: file1.py, file2.py | scope: small | group: my-feature -->`
The `group` tag is optional. Tasks sharing the same `group` value are placed on one feature branch/PR. Ungrouped tasks each get their own PR.

---

## ✅ Done

- [x] Create Cargo.toml with all Rust dependencies (pyo3 0.22, tokio, serde, serde_json, instant-distance 0.6, tokenizers 0.20, rayon 1.10, ndarray 0.16), crate-type cdylib, path rust/src/lib.rs <!-- files: Cargo.toml | scope: small | group: phase0-scaffold -->
- [x] Create pyproject.toml with maturin build backend, python-source=python, module-name=ironrace._core, dev deps (pytest, numpy, orjson, httpx, tiktoken) <!-- files: pyproject.toml | scope: small | group: phase0-scaffold -->
- [x] Create lib.rs PyO3 module entry registering ironrace._core with version() function <!-- files: rust/src/lib.rs | scope: small | group: phase0-scaffold -->
- [x] Create Python package init with _core import and version re-export <!-- files: python/ironrace/__init__.py, python/ironrace/_core.pyi | scope: small | group: phase0-scaffold -->
- [x] Verify build: maturin develop --release, import ironrace, print version <!-- scope: small | group: phase0-scaffold -->
- [x] Implement vector.rs — HNSW index via instant-distance crate, VectorIndex PyO3 class with build(vectors, ef_construction), search(query, top_k), len() methods, Point trait for cosine similarity <!-- files: rust/src/vector.rs, rust/src/lib.rs | scope: medium | group: phase1-vector-json -->
- [x] Implement json_fast.rs — serde_json parse/serialize exposed to Python via PyO3 Bound API, recursive Value<->PyObject conversion <!-- files: rust/src/json_fast.rs, rust/src/lib.rs | scope: medium | group: phase1-vector-json -->
- [x] Update _core.pyi with VectorIndex class, parse_json, serialize_json type stubs <!-- files: python/ironrace/_core.pyi | scope: small | group: phase1-vector-json -->
- [x] Write tests for vector search — correct nearest neighbor, edge cases (empty, top_k > n, single vector, high dimensions) <!-- files: tests/test_vector.py | scope: small | group: phase1-vector-json -->
- [x] Write tests for JSON bridge — roundtrip fidelity for dicts, lists, nested structures, unicode, null, large numbers <!-- files: tests/test_json.py | scope: small | group: phase1-vector-json -->
- [x] Implement tokenizer.rs — HuggingFace tokenizers crate with OnceCell init, count_tokens and truncate_to_budget with sentence-boundary binary search, fast approximate fallback <!-- files: rust/src/tokenizer.rs, rust/src/lib.rs | scope: medium | group: phase2-tokenizer-assembler -->
- [x] Implement assembler.rs — AssemblyResult PyO3 class, assemble_prompt function with template interpolation, per-section token budgets, calls crate::tokenizer internally <!-- files: rust/src/assembler.rs, rust/src/lib.rs | scope: medium | group: phase2-tokenizer-assembler -->
- [x] Update _core.pyi with count_tokens, truncate_to_budget, assemble_prompt, AssemblyResult stubs <!-- files: python/ironrace/_core.pyi | scope: small | group: phase2-tokenizer-assembler -->
- [x] Write tests for tokenizer — reasonable counts, budget enforcement, sentence boundary preservation <!-- files: tests/test_tokenizer.py | scope: small | group: phase2-tokenizer-assembler -->
- [x] Write tests for assembler — interpolation correctness, budget enforcement, sections_truncated metadata, edge cases <!-- files: tests/test_assembler.py | scope: small | group: phase2-tokenizer-assembler -->
- [x] Implement pipeline.rs — Tokio DAG executor, JSON in/out, topological sort, parallel node execution, single bridge crossing design <!-- files: rust/src/pipeline.rs, rust/src/lib.rs | scope: large | group: phase3-pipeline-sdk -->
- [x] Create types.py — TokenBudget, Document, Feature, ContextConfig dataclasses, VectorSearch/APIFetch descriptor sentinels <!-- files: python/ironrace/types.py | scope: small | group: phase3-pipeline-sdk -->
- [x] Create decorators.py — @agent, @context, @pipeline decorators with global registry, compile-on-first-call, sync+async support <!-- files: python/ironrace/decorators.py | scope: medium | group: phase3-pipeline-sdk -->
- [x] Create compiler.py — introspect registry, build dependency DAG from context descriptors, serialize to JSON for pipeline.rs <!-- files: python/ironrace/compiler.py | scope: medium | group: phase3-pipeline-sdk -->
- [x] Create router.py — async LLM router using httpx.AsyncClient, Claude/OpenAI/local dispatch, retries with exponential backoff, rate limit handling <!-- files: python/ironrace/router.py | scope: medium | group: phase3-pipeline-sdk -->
- [x] Update __init__.py with full public API re-exports (agent, context, pipeline, TokenBudget, Document, etc.) <!-- files: python/ironrace/__init__.py | scope: small | group: phase3-pipeline-sdk -->
- [x] Update _core.pyi with execute_pipeline and all remaining Rust stubs <!-- files: python/ironrace/_core.pyi | scope: small | group: phase3-pipeline-sdk -->
- [x] Write tests for pipeline executor — DAG execution, parallelism verification, dependency ordering <!-- files: tests/test_pipeline.py | scope: medium | group: phase3-pipeline-sdk -->
- [x] Write tests for decorators — registry population, metadata extraction, compile-on-first-call behavior <!-- files: tests/test_decorators.py | scope: medium | group: phase3-pipeline-sdk -->
- [x] Create baseline_python.py — adapt existing ironrace_benchmark.py as pure Python reference <!-- files: benchmarks/baseline_python.py | scope: small | group: phase4-benchmarks -->
- [x] Create bench_context_prep.py — head-to-head pure Python vs ironrace._core, all 5 operations + full pipeline, pitch-deck formatted output <!-- files: benchmarks/bench_context_prep.py | scope: medium | group: phase4-benchmarks -->
- [x] Create bench_vector_search.py — isolated vector benchmark at 1K/10K scales, build vs query time <!-- files: benchmarks/bench_vector_search.py | scope: small | group: phase4-benchmarks -->
- [x] Create bench_at_scale.py — 100/1000/10000 concurrent pipelines, wall-clock + per-invocation CPU, cost projections <!-- files: benchmarks/bench_at_scale.py | scope: medium | group: phase4-benchmarks -->
- [x] Create startup_evaluator.py — 5-agent flagship demo with shared vector search, parallel execution, timing breakdown, --dry-run/--live flags <!-- files: examples/startup_evaluator.py | scope: large | group: phase4-examples -->
- [x] Create rag_chatbot.py — simple RAG agent with vector index + token budget <!-- files: examples/rag_chatbot.py | scope: small | group: phase4-examples -->
- [x] Create batch_research.py — 100 parallel research pipelines via ThreadPoolExecutor <!-- files: examples/batch_research.py | scope: small | group: phase4-examples -->
- [x] Write docs/PROBLEM.md — business case, benchmark data, Pydantic/Polars/Ruff trend <!-- files: docs/PROBLEM.md | scope: medium | group: phase5-docs -->
- [x] Write docs/ARCHITECTURE.md — two-phase model, PyO3 bridge-once, Tokio parallelism, token budgeting, memory model <!-- files: docs/ARCHITECTURE.md | scope: medium | group: phase5-docs -->
- [x] Write docs/QUICKSTART.md — 5-min guide <!-- files: docs/QUICKSTART.md | scope: small | group: phase5-docs -->
- [x] Write docs/BENCHMARKS.md — methodology, sample output, reproduction steps <!-- files: docs/BENCHMARKS.md | scope: small | group: phase5-docs -->
- [x] Write README.md — overview, install, 30-second example, benchmark highlight <!-- files: README.md | scope: small | group: phase5-docs -->
- [x] Add py.typed marker, complete _core.pyi stubs <!-- files: python/ironrace/py.typed, python/ironrace/_core.pyi | scope: small | group: phase5-polish -->
