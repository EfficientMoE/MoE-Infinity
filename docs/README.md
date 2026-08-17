# Documentation Hub

This hub is the main index for the longer guides, grouped by audience so readers can jump to the right details quickly. If you are choosing a model, start with the compatibility matrix, then open the family guide for any special cases. For runtime behavior, use DFlash for sync speculative decode, OpenAI serving for the async server and cancellation flow, and Architecture for the public vs internal boundary.
The configuration and environment reference pages are sourced from the code, so the field names and defaults stay aligned with the implementation.

## Users

Users are people who want to pick a model, check the compatibility matrix, and plan memory before reading deeper runtime details.

- [Model compatibility matrix](./model-compatibility.md)
- [Configuration and memory planning](./configuration.md)
- [DFlash speculative decode](./dflash.md)
- [GLM-5.2 guide](glm-5.2.md)
- [DeepSeek-V4-Flash guide](../moe_infinity/models/deepseek_v4/README.md)
- [ContextPilot](./contextpilot/README.md)

## Operators

Operators are people who run the OpenAI server and need runtime controls for deployment and support.

- [OpenAI serving](./serving.md)
- [Environment variables](./environment-variables.md)
- [Single-server multi-GPU](./multi-gpu.md)
- [Troubleshooting](./troubleshooting.md)

## Contributors

Contributors are people who change code, docs, or packaging and need the technical references before sending a patch, especially the architecture map and API stability table.

- [Architecture](../ARCHITECTURE.md)
- [Benchmarking catalog and runbooks](./benchmarking.md)
- [Benchmark reproduction](./benchmark_reproduction.md)
- [Expert I/O microbench runbook](../benchmarks/expert_io_microbench/README.md)
- [C++ interface](./cpp-interface-spec.md)
- [Contributing](../CONTRIBUTING.md)

## Project History

Project History is for people who want the release record and the process for cutting new releases.

- [Changelog](../CHANGELOG.md)
- [Release process](../RELEASE.md)
