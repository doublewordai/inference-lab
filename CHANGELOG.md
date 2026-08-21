# Changelog

## [1.0.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.10.0...inference-lab-v1.0.0) (2026-08-21)


### ⚠ BREAKING CHANGES

* **memory:** KV movement policies (reactive | oracle), DP-attention ranks as workers, radix-tree KV state ([#103](https://github.com/doublewordai/inference-lab/issues/103))
* **memory:** KV memory graph — stores, links, junctions, max-min paths; hand-offs on the graph ([#93](https://github.com/doublewordai/inference-lab/issues/93))
* **catalog:** hardware presets b200-datasheet, gh200-120 and gh200-96 are gone (use b200 / gh200); b200 moves from 180 GiB / 7.7 TB/s to 192 GB / 8 TB/s, b300 to 288 GB, h100 to 80 GB.
* ClusterSpec has no comms field; HardwareConfig gains fabric; multi-GPU deployments on fabric-less hardware are rejected; ep must divide tp; tp > 1 results move (collectives now cost).
* config file format; Config::from_file removed; --workload required for sim.
* configs reference catalog entries (hardware = "b200", model = "<slug>"), unknown fields are rejected, [simulation]/draft_cost_frac/SlidingWindowModel/Node removed, --output JSON is the nested MetricsSummary shape. Per-commit history on branch review/cleanup.

### Features

* collective fabric on hardware presets; tp/ep/dp-attention semantics; prod layouts in configs ([#80](https://github.com/doublewordai/inference-lab/issues/80)) ([37d6dbf](https://github.com/doublewordai/inference-lab/commit/37d6dbf04ba354bdfe0b65be20affa1c2a00b3c8))
* **memory:** backup = on_land forwards landed writes to the next tier ([#127](https://github.com/doublewordai/inference-lab/issues/127)) ([bf33b42](https://github.com/doublewordai/inference-lab/commit/bf33b426d1ddf38010fe8dbcac84fc30538cbfbe))
* **memory:** hit_refresh = first_tier — an HBM prefix hit re-stamps the first tier's copy ([#129](https://github.com/doublewordai/inference-lab/issues/129)) ([80564e1](https://github.com/doublewordai/inference-lab/commit/80564e162b6bb4778dd2d8c54ad5189dc7898202))
* **memory:** KV memory graph — stores, links, junctions, max-min paths; hand-offs on the graph ([#93](https://github.com/doublewordai/inference-lab/issues/93)) ([7929f77](https://github.com/doublewordai/inference-lab/commit/7929f77bcc467c1578a26c9a34e9c9d1a1ace397))
* **memory:** KV movement policies (reactive | oracle), DP-attention ranks as workers, radix-tree KV state ([#103](https://github.com/doublewordai/inference-lab/issues/103)) ([10c4472](https://github.com/doublewordai/inference-lab/commit/10c4472871c427266572a83431b4a49ed0eafcae))
* **memory:** model staged HiCache reads ([#132](https://github.com/doublewordai/inference-lab/issues/132)) ([24d4011](https://github.com/doublewordai/inference-lab/commit/24d4011273fd1be1413a38f262ae40353f2d5541))
* **memory:** promote_fill = through — a promotion refills the tiers above its source ([#130](https://github.com/doublewordai/inference-lab/issues/130)) ([81117d4](https://github.com/doublewordai/inference-lab/commit/81117d40cb038c43ebd775cf5e18a6300f08194a))
* **memory:** write policies, inclusive tiers, store eviction policies, memory metrics ([#96](https://github.com/doublewordai/inference-lab/issues/96)) ([37d5ba4](https://github.com/doublewordai/inference-lab/commit/37d5ba49bc7838cd66f9c93aa8ddfbbe200893b4))
* **router:** pluggable request router across replicas; LRU KV block recycling; disagg hand-off fixes ([#90](https://github.com/doublewordai/inference-lab/issues/90)) ([11172e1](https://github.com/doublewordai/inference-lab/commit/11172e106192135adb3ce994f2b83a5fc63b52b0))
* **workload:** session workloads — re-entering request chains from a trace ([#89](https://github.com/doublewordai/inference-lab/issues/89)) ([0fbc6ae](https://github.com/doublewordai/inference-lab/commit/0fbc6ae4d5ced78f36a04a08a55b0ecb361c16b6))


### Bug Fixes

* closed-loop jitter stall, Docker build inputs, CI test workflow ([a2072e0](https://github.com/doublewordai/inference-lab/commit/a2072e0aa8804d040c22225f7d7e733d7ee119be))
* **deps:** update rand to 0.10, rand_distr to 0.6, getrandom to 0.4 ([#36](https://github.com/doublewordai/inference-lab/issues/36)) ([eeb2b2b](https://github.com/doublewordai/inference-lab/commit/eeb2b2b3cdb829816505ee98a919ba326bbd1e7e))
* **deps:** update rust crate tabled to 0.21 ([#92](https://github.com/doublewordai/inference-lab/issues/92)) ([4adc4f8](https://github.com/doublewordai/inference-lab/commit/4adc4f88c64c124a4be3aeb80d3e2bf00ea4cae0))
* **deps:** update rust crate tokenizers to 0.23 ([#94](https://github.com/doublewordai/inference-lab/issues/94)) ([1ae6c91](https://github.com/doublewordai/inference-lab/commit/1ae6c917b6280c0c7edd74cd94d0144a4272549a))
* **deps:** update rust crate tower-http to 0.7 ([#95](https://github.com/doublewordai/inference-lab/issues/95)) ([7d16268](https://github.com/doublewordai/inference-lab/commit/7d1626818958a85a6b845fc96a5dfa7ee408c6b6))
* **engine:** EP with TP attention prices the FFN all-reduce, not all-to-alls ([#85](https://github.com/doublewordai/inference-lab/issues/85)) ([107b0c2](https://github.com/doublewordai/inference-lab/commit/107b0c22f04b89e4a8354b9a5af69f327b8bce22))
* modelling review — causal prefill attention, absorbed MLA decode, hybrid-SWA prefix hits, admission rule, disagg TTFT, opt-in step calibration ([#100](https://github.com/doublewordai/inference-lab/issues/100)) ([1e87653](https://github.com/doublewordai/inference-lab/commit/1e876536c70a94856f72801be17740ce71a31130))
* **model:** price DSA indexer scoring at its own precision ([#126](https://github.com/doublewordai/inference-lab/issues/126)) ([236c0c0](https://github.com/doublewordai/inference-lab/commit/236c0c0182cc33a8f37b0dedd17ffced951a9b6b))
* **scheduler:** landed promotions must not starve running requests ([#119](https://github.com/doublewordai/inference-lab/issues/119)) ([0d50691](https://github.com/doublewordai/inference-lab/commit/0d506917a11b745fbdb23c3216ebbe58c3794362))


### Performance Improvements

* cache request radix leaves ([#113](https://github.com/doublewordai/inference-lab/issues/113)) ([77aa127](https://github.com/doublewordai/inference-lab/commit/77aa1270c51ff0107fb77c799ebd8b135f9777bf))
* collect the time series only when asked ([#117](https://github.com/doublewordai/inference-lab/issues/117)) ([6d573c0](https://github.com/doublewordai/inference-lab/commit/6d573c0374f21af4bdc5803e51c67881734a4f9d))
* optimize release builds and allocator ([#115](https://github.com/doublewordai/inference-lab/issues/115)) ([b584a96](https://github.com/doublewordai/inference-lab/commit/b584a96b5dff2fd93abd01331cf2052a74168d73))
* resolve router prefixes once per arrival ([#110](https://github.com/doublewordai/inference-lab/issues/110)) ([a71757e](https://github.com/doublewordai/inference-lab/commit/a71757e9810445806db1ce23de5a6817e4e92cc2))
* skip idle DP-attention ranks ([#108](https://github.com/doublewordai/inference-lab/issues/108)) ([9e7a396](https://github.com/doublewordai/inference-lab/commit/9e7a3961f0ca2e17990311146b4a47d39d404b00))


### Miscellaneous Chores

* **catalog:** one datasheet preset per GPU ([#88](https://github.com/doublewordai/inference-lab/issues/88)) ([0c87c33](https://github.com/doublewordai/inference-lab/commit/0c87c33b6ce69f9c4b36fa71b43ca9cd8733716a))


### Code Refactoring

* exact KV accounting, composable ModelSpec, shipped hardware/model catalog ([#74](https://github.com/doublewordai/inference-lab/issues/74)) ([7975527](https://github.com/doublewordai/inference-lab/commit/7975527ecd69868cdf19fe06c1942f5e195f6607))
* one model config per deployment with [hardware.&lt;name&gt;] entries; workloads split out ([#77](https://github.com/doublewordai/inference-lab/issues/77)) ([1788006](https://github.com/doublewordai/inference-lab/commit/17880069ce408cd375a3f4ab8bd6640a54b22333))

## [0.10.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.9.0...inference-lab-v0.10.0) (2026-08-12)


### Features

* OpenAI compat resume target ([#72](https://github.com/doublewordai/inference-lab/issues/72)) ([0654c03](https://github.com/doublewordai/inference-lab/commit/0654c037d9cb7c4a5c0cff9cd95b887c5f430115))

## [0.9.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.8.2...inference-lab-v0.9.0) (2026-08-11)


### Features

* fault injection for midstream error testing ([#69](https://github.com/doublewordai/inference-lab/issues/69)) ([c817058](https://github.com/doublewordai/inference-lab/commit/c817058f36fc73c6044cb8852f5173fd378aba9c))

## [0.8.2](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.8.1...inference-lab-v0.8.2) (2026-08-06)


### Bug Fixes

* report chat prompt_tokens with simulated template overhead ([#67](https://github.com/doublewordai/inference-lab/issues/67)) ([1601616](https://github.com/doublewordai/inference-lab/commit/16016168a414dfea065b569a270f24ba7bd5c102))

## [0.8.1](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.8.0...inference-lab-v0.8.1) (2026-07-31)


### Bug Fixes

* count tool definitions and tool_calls in serve prompt_tokens ([#64](https://github.com/doublewordai/inference-lab/issues/64)) ([aaa8ce9](https://github.com/doublewordai/inference-lab/commit/aaa8ce90fd7e1180f6acf225729fa32dd17a1072))

## [0.8.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.7.1...inference-lab-v0.8.0) (2026-07-28)


### Features

* echo directive ([#62](https://github.com/doublewordai/inference-lab/issues/62)) ([dcc7f2d](https://github.com/doublewordai/inference-lab/commit/dcc7f2d53e70ff4978b1a6ee6be0e7a1217065ab))

## [0.7.1](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.7.0...inference-lab-v0.7.1) (2026-07-15)


### Bug Fixes

* allow content array blocks as input ([#60](https://github.com/doublewordai/inference-lab/issues/60)) ([2e51610](https://github.com/doublewordai/inference-lab/commit/2e516101f20c643b541abb107f3636f31164e033))

## [0.7.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.6.2...inference-lab-v0.7.0) (2026-06-22)


### Features

* model KV cache hierarchy, cascade attention, and async promotions ([#44](https://github.com/doublewordai/inference-lab/issues/44)) ([59e7ac1](https://github.com/doublewordai/inference-lab/commit/59e7ac1aad90dd9782bd7271cc7cd7be52af23d7))
* **spec:** speculative-decoding simulator + calibration package + figure tooling ([#53](https://github.com/doublewordai/inference-lab/issues/53)) ([299a718](https://github.com/doublewordai/inference-lab/commit/299a718fad5124c420392eb7a81d3c3de0153f7d))


### Bug Fixes

* **deepseek-v4-pro:** correct per-layer-class counts ([#49](https://github.com/doublewordai/inference-lab/issues/49)) ([5ed8b94](https://github.com/doublewordai/inference-lab/commit/5ed8b94621a4498cebaceba0eb93831598534891))

## [0.6.2](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.6.1...inference-lab-v0.6.2) (2026-03-25)


### Bug Fixes

* gate streaming usage on include_usage ([#42](https://github.com/doublewordai/inference-lab/issues/42)) ([320af44](https://github.com/doublewordai/inference-lab/commit/320af446bb327e9a63e1f3a1785aab56980bcccf))

## [0.6.1](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.6.0...inference-lab-v0.6.1) (2026-03-25)


### Bug Fixes

* streaming usage reporting ([#40](https://github.com/doublewordai/inference-lab/issues/40)) ([088cf34](https://github.com/doublewordai/inference-lab/commit/088cf34cf8301abc73694780549fcee8917e904a))

## [0.6.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.5.0...inference-lab-v0.6.0) (2026-03-23)


### Features

* add /v1/completions support ([#38](https://github.com/doublewordai/inference-lab/issues/38)) ([bc3e5fb](https://github.com/doublewordai/inference-lab/commit/bc3e5fbda370da6f1acc67dc94be0822c3e2e6ba))

## [0.5.0](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.4.3...inference-lab-v0.5.0) (2026-01-24)


### Features

* multi-model serve mode with directory-based config loading ([3bbe32f](https://github.com/doublewordai/inference-lab/commit/3bbe32fc8cf5d1cdd398968519b1fab527302857))

## [0.4.3](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.4.2...inference-lab-v0.4.3) (2026-01-24)


### Bug Fixes

* build versioned Docker images from release-please workflow ([671ed78](https://github.com/doublewordai/inference-lab/commit/671ed78a22ffe150e025cb04b7f1c8e5cadd6330))

## [0.4.2](https://github.com/doublewordai/inference-lab/compare/inference-lab-v0.4.1...inference-lab-v0.4.2) (2026-01-24)


### Bug Fixes

* trigger Docker build on release events ([ea3e3ea](https://github.com/doublewordai/inference-lab/commit/ea3e3eab1217fce6cde3770847a88ebe53fe1f0b))

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
