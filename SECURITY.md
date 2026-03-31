# Security Policy

## Supported Versions

MoE-Infinity currently publishes from the `0.0.x` series (base version set in `setup.py` as `0.0.1`, with CI-generated `0.0.1dev*` builds for TestPyPI).

We provide security support for the latest maintained code as follows:

| Version | Supported |
| --- | --- |
| `main` branch (latest development) | ✅ |
| Latest `0.0.x` release on PyPI | ✅ |
| Older/unpinned snapshots and forks | ❌ |

If you are running a custom fork, please first verify whether the issue reproduces on `main`.

## Scope

This policy covers vulnerabilities in:

- `moe_infinity/` Python runtime and serving code
- Native/CUDA extensions shipped by this repository
- Official release artifacts published from this repository

Configuration mistakes or local environment issues that are not security defects should be reported through regular issues instead.

## Reporting a Vulnerability

Please report vulnerabilities **privately** through one of these channels:

1. GitHub Security Advisories (preferred):
   - https://github.com/EfficientMoE/MoE-Infinity/security/advisories/new
2. Email:
   - `moe-infinity@googlegroups.com`

**Do not** report security vulnerabilities in public GitHub issues.

If you are unsure whether something is security-related, contact us privately first and we will help classify it.

When reporting, include:

- Affected MoE-Infinity version/commit
- Environment details (OS, Python, PyTorch, CUDA, GPU)
- Clear reproduction steps or proof-of-concept
- Security impact (confidentiality, integrity, availability)
- Any suggested mitigation or patch

Optional but helpful:

- Affected model/checkpoint context
- Relevant logs or stack traces (sanitized)
- Whether the issue requires local access, authenticated access, or remote access

## Response Process and Timeline

Our maintainers will:

- Acknowledge receipt within **72 hours**
- Provide an initial triage/severity assessment within **7 days**
- Share status updates at least every **14 days** until resolution

For validated reports, we will coordinate a fix and responsible disclosure timeline based on severity and ecosystem impact.

## Triage and Resolution Process

After triage, we generally follow this flow:

1. Reproduce and assess impact
2. Develop and validate a fix
3. Prepare release/advisory notes
4. Coordinate disclosure and publish remediation guidance

For critical issues, we may prioritize out-of-band fixes and accelerated release timelines.

## Disclosure Policy

- Please allow maintainers time to investigate and patch before public disclosure.
- Once a fix is available, we may publish a security advisory with impact, affected versions, and remediation guidance.
- Please avoid sharing proof-of-concept exploit details publicly until a fix or mitigation is available.

## Non-Security Bug Reports

For non-security defects, use the public bug template:

- https://github.com/EfficientMoE/MoE-Infinity/issues/new?template=bug_report.yml

## Security Best Practices for Users

Because MoE-Infinity integrates with external model ecosystems:

- Use trusted model checkpoints and repositories only
- Review remote model code and configuration before execution
- Keep dependencies (especially PyTorch/Transformers/CUDA stack) updated
- Run inference services with least privilege and network restrictions where possible

Thank you for helping keep MoE-Infinity and its users safe.
