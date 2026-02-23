# Contributing to BloomBee

Thank you for your interest in contributing! This guide covers how to set up your development environment, follow the project's code style, run tests, and submit changes.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Development Setup

**Requirements:** Python 3.8+, Git

```bash
# 1. Fork and clone the repository
git clone https://github.com/ai-decentralized/BloomBee.git
cd BloomBee

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 3. Install the package with dev dependencies
pip install -e ".[dev]"

# 4. Install pre-commit hooks (runs formatters automatically before each commit)
pip install pre-commit
pre-commit install
```

> **Note:** The full install includes `hivemind` (fetched from Git) and `torch`, so the first install may take a few minutes.

---

## Code Style

BloomBee uses **black** for formatting and **isort** for import ordering. Both are enforced in CI.

| Tool | Version | Config |
|---|---|---|
| black | 22.3.0 | `pyproject.toml` → `[tool.black]` |
| isort | 5.10.1 | `pyproject.toml` → `[tool.isort]` |
| pylint | latest | `.pylintrc` |
| Line length | 120 | all tools |

### Formatting manually

```bash
# Format all source files
black src/bloombee tests benchmarks
isort src/bloombee tests benchmarks

# Check without modifying (what CI does)
black --check src/bloombee tests benchmarks
isort --check-only src/bloombee tests benchmarks

# Lint
pylint src/bloombee
```

### Pre-commit hooks

If you installed pre-commit (step 4 above), black and isort run automatically on every `git commit`. To run all hooks manually:

```bash
pre-commit run --all-files
```

---

## Running Tests

> **Important:** Most tests are **integration tests** that require a live BloomBee swarm. You need to start a bootstrap node and at least one worker before running them (see the [Quick Start](README.md#quick-start) in the README).

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `INITIAL_PEERS` | Yes | Multiaddress(es) of the bootstrap node |
| `MODEL_NAME` | Yes | HuggingFace model ID served by your swarm |
| `REF_NAME` | No | HuggingFace model ID for reference comparison |
| `ADAPTER_NAME` | No | LoRA adapter name for PEFT tests |

### Running the tests

```bash
export INITIAL_PEERS="/ip4/YOUR_IP/tcp/31340/p2p/Qm..."
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export REF_NAME="meta-llama/Llama-2-7b-hf"

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_full_model.py -v

# Run a specific test
pytest tests/test_full_model.py::test_full_model_exact_match -xvs
```

---

## Submitting Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**, keeping commits focused and atomic.

3. **Ensure your code is formatted** — pre-commit handles this automatically, or run `black` and `isort` manually.

4. **Push your branch** and open a pull request against `main`.

5. Fill out the pull request template with a description of what you changed and why.

### What makes a good PR

- Focused on a single concern (one feature, one bug fix)
- Includes a clear description of the problem being solved
- Does not introduce unnecessary dependencies
- Passes all CI checks (formatting, linting)

---

## Reporting Issues

Use the GitHub issue tracker:

- **Bug reports:** Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Please include your Python version, OS, GPU/CUDA info, and a minimal reproduction.
- **Feature requests:** Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).
- **Questions:** Start a discussion in the [Discord server](https://discord.gg/Ypexx2rxt9).

---

## Project Structure (quick reference)

```
src/bloombee/
├── client/         # Client-side inference sessions and routing
├── server/         # Server RPC handler, backend, memory cache
├── models/         # Per-architecture model classes (llama, bloom, falcon, mixtral)
├── cli/            # Entry points: run_dht, run_server
├── utils/          # Auto-config, DHT utilities
└── flexgen_utils/  # FlexGen offloading integration

tests/              # Integration test suite
benchmarks/         # Inference, forward, and training benchmarks
examples/           # Jupyter notebooks (prompt tuning)
```
