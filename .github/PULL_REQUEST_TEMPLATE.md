## Description
Briefly describe your changes.

## Motivation
Explain why this change is needed and what problem it solves.
If it fixes an issue, link it (e.g., `close #123`).

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Documentation impact
Check every item that applies.

- [ ] README discovery updated for any user-facing feature or behavior change.
- [ ] Authoritative feature, config, or API guide updated.
- [ ] Model compatibility, examples, architecture, or benchmark docs updated if this change touches them.
- [ ] CHANGELOG `Unreleased` updated.
- [ ] No docs impact. Explain why in the description or motivation.

Do not check `No docs impact` if any of the update boxes above are checked unless the description explains why the existing docs already cover the change.

## Performance / support evidence
Use this section for any change that affects performance, supportability, or model behavior. A checked box alone is not enough.

- [ ] Not applicable — explain in the PR description why no performance/support evidence is needed.

| Field | Fill in when applicable |
| --- | --- |
| Model / checkpoint | <!-- e.g. `deepseek-ai/DeepSeek-V2-Lite-Chat` --> |
| Hardware | <!-- e.g. `1× RTX 4090, 24 GB` --> |
| Software versions | <!-- e.g. `Python 3.12`, `torch 2.8.0+cu128`, `transformers 5.12` --> |
| Workload | <!-- e.g. `128 prompts`, `batch size 8`, `4K context` --> |
| Baseline | <!-- e.g. `main@<sha>`, previous release, or alternate path --> |
| Measured result (if applicable) | <!-- e.g. `TTFT 1.8s → 1.2s`, `throughput 210 tok/s → 256 tok/s` --> |
| Limitations / validation scope | <!-- e.g. `single GPU only`, `no multi-node validation` --> |

If this change is applicable, fill every row with concrete values. Do not leave the section satisfied by the checkbox alone.

## Checklist
- [ ] I have read the [CONTRIBUTION](https://github.com/EfficientMoE/MoE-Infinity/blob/main/CONTRIBUTING.md) guide.
- [ ] I have updated the tests (if applicable).
- [ ] I have filled out the documentation impact section above.
