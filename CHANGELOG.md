# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-24
### Added
- Initial public release of **freegrad**.
- Core: registry for custom gradient rules; context manager to apply rules.
- Wrappers: activation module with alternative backward transforms.
- Built-in transforms: `d(ReLU)`, `d(Linear)`, `rectangular`, `triangular`, `scale`, `clip_norm`, `noise`, `centralize`.
- Jamming transforms: `full_jam`, `positive_jam`, `rectangular_jam`.
- Basic docs: README, CONTRIBUTING, LICENSE.
- CI scaffold and developer tooling (pytest, black, ruff, mypy, pre-commit).
