# CLAUDE.md

## Project

ironrace: Rust-powered context engine for AI agent pipelines. Rust core compiled to a Python extension via PyO3/maturin.

## Build & Run

```bash
# Create venv and build the extension
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop --release

# Run tests
pytest tests/ -v

# Integration tests (llama-index)
pip install llama-index-core
pip install -e integrations/llama-index
cd integrations/llama-index && IS_TESTING=true pytest tests/ -v
```

## Codebase Layout

- **Rust core**: `rust/src/` — PyO3 extension module (`_core`)
- **Python wrapper**: `python/ironrace/` — public API, decorators, types, compiler
- **Tests**: `tests/` — pytest suite (assembler, decorators, json, pipeline, tokenizer, vector)
- **Integrations**: `integrations/llama-index/` — LlamaIndex integration package
- **CI**: `.github/workflows/ci.yml` — tests on push/PR to main
- **Release**: `.github/workflows/release.yml` — builds wheels (linux, macos, windows) and publishes to PyPI on tag push

## Rust Conventions

- Extension module name is `_core` (crate-type cdylib)
- PyO3 0.22 with `extension-module` feature
- Keep Rust code in `rust/src/`, not `src/`

## Python Conventions

- Python source root is `python/` (configured in pyproject.toml `tool.maturin.python-source`)
- Public API is `ironrace` package; Rust bindings accessed via `ironrace._core`
- Type stubs in `python/ironrace/_core.pyi`; update when adding/changing Rust-exposed functions
- Requires Python >= 3.11

## Testing Requirements

- Run `pytest tests/ -v` before committing
- Performance tests must use generous thresholds; CI runners are slower than local machines
- Do not add tests that make network calls

## Release

- Wheels built via `PyO3/maturin-action` with `--find-interpreter`
- Release triggered by pushing a `v*` tag
- PyPI publish uses trusted publishing (OIDC, no API tokens)

## Documentation Rules

- When a feature or public API changes, update all affected docs in the same commit (README.md, docs/*.md, examples/)
- New Rust functions exposed via PyO3 require an updated type stub in `python/ironrace/_core.pyi`
- Keep docs/ARCHITECTURE.md in sync with any structural changes to the Rust or Python layers

## Test Coverage Rules

- All tests must pass (`pytest tests/ -v`) before committing code
- New features require both positive tests (happy path) and negative tests (invalid input, error conditions, edge cases)
- Integration tests (`integrations/llama-index/tests/`) must also pass if integration code was touched
- Performance tests must use generous thresholds; CI runners are slower than local machines

## Security Rules

- Never commit secrets, API keys, or credentials into the repository
- Any `unsafe` Rust code must include a `// SAFETY:` comment explaining why the block is sound
- Do not expose raw internal error messages to Python callers; wrap them in meaningful error types

## Git Workflow

- Use conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`
- Keep commits focused — one logical change per commit
- PRs should target `main` and include a description of what changed and why

## Writing Style

- Documentation should be concise and direct; no filler words or marketing language
- Code comments explain "why", not "what"
- Prefer concrete examples over abstract descriptions in docs
