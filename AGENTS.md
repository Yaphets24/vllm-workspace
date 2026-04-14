# Repository Guidelines

## Project Structure & Module Organization

This repository is a local integration workspace centered on `vllm-ascend/`.

- `vllm-ascend/`: the primary development target. It contains the Ascend plugin source in `vllm_ascend/`, unit tests in `tests/ut/`, end-to-end tests in `tests/e2e/`, docs in `docs/`, and helper scripts in `tools/`.
- `vllm/`: a read-only upstream reference. Use it only to inspect APIs, understand dependency behavior, and check compatibility assumptions.

The root itself has no package, build, or test logic. All contributor changes in this workspace should land in `vllm-ascend/`. Do not modify files under `vllm/`.

## Build, Test, and Development Commands

Run commands from `vllm-ascend/`. Use `vllm/` only for code reading and behavior comparison.

- `cd vllm-ascend && pip install -r requirements-dev.txt`: install plugin development dependencies.
- `cd vllm-ascend && pytest -sv tests/ut`: run Ascend unit tests.
- `cd vllm-ascend && pytest -sv tests/e2e/singlecard`: run plugin E2E coverage on supported hardware.
- `cd vllm-ascend && ruff check vllm_ascend/ && ruff format vllm_ascend/`: lint and format plugin code.

Use targeted test paths for faster iteration, for example `pytest -sv tests/ut/test_ascend_config.py`.

## Coding Style & Naming Conventions

Python uses 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and `ALL_UPPER_CASE` for constants. Follow existing patterns in `vllm-ascend` first. Keep environment variable definitions centralized in `vllm_ascend/envs.py`, and avoid scattering new env var names across the codebase.

## Testing Guidelines

Name tests `test_*.py`. In `vllm-ascend`, mirror source paths when adding unit tests, for example `vllm_ascend/worker/worker.py` to `tests/ut/worker/test_worker.py`. New features need tests, bug fixes need regression coverage, and NPU-specific behavior should be validated in `tests/e2e/` when applicable. When upstream behavior matters, read `vllm/` as the compatibility baseline but keep implementation and tests in `vllm-ascend/`.

## Commit & Pull Request Guidelines

This root repo mainly tracks workspace imports, so keep all contributor changes scoped to `vllm-ascend/`. Use `git commit -s` for DCO sign-off. For `vllm-ascend`, follow Conventional Commits for commit messages and use PR titles in the form `[Type][Module] Description`. If you find an upstream issue while working, document it in the PR or issue discussion rather than editing `vllm/` here.
