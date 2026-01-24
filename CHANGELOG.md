# Changelog

## [0.4.1](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.4.0...inference-lab-v0.4.1) (2026-01-24)


### Bug Fixes

* match release-please tag format in Docker workflow ([4a41b05](https://github.com/doublewordai/inference-lab/commit/4a41b05b11af6fd6f152837740b1ef090e29bdc6))

## [0.4.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.3.1...inference-lab-v0.4.0) (2026-01-24)


### Features

* add OpenAI-compatible serve mode with Docker support ([2d12260](https://github.com/doublewordai/inference-lab/commit/2d12260f0391154a2f7ac4568cd62514e1b1a8f5))

## [0.3.1](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.3.0...inference-lab-v0.3.1) (2025-12-11)


### Bug Fixes

* **deps:** update rust crate tokenizers to 0.22 ([3d14b09](https://github.com/doublewordai/inference-lab/commit/3d14b09aaaafdafdea10d1bc3d128d818e2c5418))

## [0.3.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.2.0...inference-lab-v0.3.0) (2025-12-11)


### Features

* add SOF/SIF/STF/LIF/LOF/LTF scheduling ([86ddad5](https://github.com/doublewordai/inference-lab/commit/86ddad5af805105e3820a74ab402115a94cf1269))
* bump version ([53df6e6](https://github.com/doublewordai/inference-lab/commit/53df6e67b9cb2f62e78294e5ac62395df0774a0e))
* real datasets ([935331d](https://github.com/doublewordai/inference-lab/commit/935331dc106fb74d29488e5ff77b6ee5d01b6a77))


### Bug Fixes

* enable closed loop + dataset mode ([0a64745](https://github.com/doublewordai/inference-lab/commit/0a64745d77dfc479a2b6c1ad82929c55fdf28b9b))
* optimizations ([f8d76be](https://github.com/doublewordai/inference-lab/commit/f8d76be4342dfe1a3856bf3d19ed3c174e41441d))
* proper prefix caching modelling ([cc6947b](https://github.com/doublewordai/inference-lab/commit/cc6947b72846ebecde8abc4c0a9f1726612ebc9c))
* update readme ([6316efe](https://github.com/doublewordai/inference-lab/commit/6316efe9a565ca2b2c0ede35b33d4dd6ca52ac5f))

## [0.2.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.1.1...inference-lab-v0.2.0) (2025-12-05)


### Features

* add minimum latencies ([b9c603c](https://github.com/doublewordai/inference-lab/commit/b9c603c3f9c8163e3746d00e88c6a04faded59b6))
* configure for GitHub Packages publishing ([a383ba9](https://github.com/doublewordai/inference-lab/commit/a383ba9b3ccc888b14adc5e2864b267be29b7382))
* minimum latencies ([bf065e3](https://github.com/doublewordai/inference-lab/commit/bf065e371c1d012634e5af8b624a710979c9daf4))
* publish to npm registry ([97519b0](https://github.com/doublewordai/inference-lab/commit/97519b0c5747209f7829483ff25a0079dfcbfa9a))


### Bug Fixes

* **deps:** update rust crate getrandom to 0.3 ([3027959](https://github.com/doublewordai/inference-lab/commit/3027959072348612dcc075d40750b2c53368a475))
* **deps:** update rust crate getrandom to 0.3 ([5bc89c1](https://github.com/doublewordai/inference-lab/commit/5bc89c12e25f3d32c47e552e62a297e35f2d908c))
* remove unsupported release-please parameters ([2949d4d](https://github.com/doublewordai/inference-lab/commit/2949d4d1e0bdc27be1b4000ae52f7a7432fe056a))
* sync release-please with published versions ([ac9e8e6](https://github.com/doublewordai/inference-lab/commit/ac9e8e6f38b68fdd3948623fb6e7fa42557ece9c))
* update imports to use inference-lab crate name ([fcc6304](https://github.com/doublewordai/inference-lab/commit/fcc630476b684fb0e5ca769fbe76f8eb202bc278))
* update repository URLs to doublewordai organization ([2a52eb1](https://github.com/doublewordai/inference-lab/commit/2a52eb1e4b8898fb4b2fa4d0504abffd58fa8461))

## [0.1.1](https://github.com/doublewordai/inference-lab/compare/v0.1.0...v0.1.1) (2025-12-05)


### Bug Fixes

* sync release-please with published versions ([ac9e8e6](https://github.com/doublewordai/inference-lab/commit/ac9e8e6f38b68fdd3948623fb6e7fa42557ece9c))

## 0.1.0 (2025-12-05)


### Features

* configure for GitHub Packages publishing ([a383ba9](https://github.com/doublewordai/inference-lab/commit/a383ba9b3ccc888b14adc5e2864b267be29b7382))
* publish to npm registry ([97519b0](https://github.com/doublewordai/inference-lab/commit/97519b0c5747209f7829483ff25a0079dfcbfa9a))


### Bug Fixes

* update imports to use inference-lab crate name ([fcc6304](https://github.com/doublewordai/inference-lab/commit/fcc630476b684fb0e5ca769fbe76f8eb202bc278))
* update repository URLs to doublewordai organization ([2a52eb1](https://github.com/doublewordai/inference-lab/commit/2a52eb1e4b8898fb4b2fa4d0504abffd58fa8461))

## [0.1.0](https://github.com/doublewordai/inference-lab/releases/tag/v0.1.0) (2025-12-05)

### Features

* Initial release of Inference Lab
* High-performance LLM inference simulator
* Support for multiple scheduling policies (FCFS, Priority, SJF)
* Chunked prefill simulation
* KV cache management
* Workload generation (Poisson, Gamma, closed-loop)
* WebAssembly support for browser usage
* CLI tool for command-line simulation
* Published to crates.io and npm
