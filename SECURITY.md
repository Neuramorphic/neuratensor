# Security & Verification

## Supply Chain Security

### Current State (v4.0.2)

⚠️ **Limited Provenance**: The current PyPI releases (v4.0.0 - v4.0.2) were published using a manual API token, which does not generate cryptographic provenance attestations. While the packages are built from this repository, there is no automated chain of custody verification from source code to published package.

### Planned Improvements (v4.1.0+)

We are implementing **Trusted Publishing** with GitHub Actions:

- ✅ GitHub Actions workflow for automated builds (see `.github/workflows/publish.yml`)
- 🔄 PyPI Trusted Publisher configuration (requires repository owner access)
- 🔄 Automated provenance attestation generation
- 🔄 SLSA Build Level 3 compliance

Once enabled, every release will have a cryptographically signed attestation that proves:
- Which GitHub repository it came from
- Which commit/tag triggered the build
- That the build ran in GitHub's isolated infrastructure
- That no human had direct access to PyPI credentials

You can track progress in issue [#TBD].

## Verification Steps

Until Trusted Publishing is enabled, we recommend:

1. **Verify package integrity:**
   ```bash
   pip download neuratensor==4.0.2
   sha256sum neuratensor-4.0.2-py3-none-any.whl
   # Compare against SHA256 in release notes
   ```

2. **Build from source:**
   ```bash
   git clone https://github.com/neuramorphic/neuratensor.git
   cd neuratensor
   git checkout v4.0.2
   python -m build
   pip install dist/*.whl
   ```

3. **Review the code:** All source code is open (Apache 2.0) and auditable.

## Benchmark Claims

Benchmark results in the README are **format examples**, not verified performance claims. Actual performance varies significantly by:

- Hardware (CPU model, GPU, memory bandwidth)
- PyTorch version and compilation flags
- Input data characteristics
- Model size and configuration

**For production evaluation:** Run benchmarks on your target hardware:

```bash
neuratensor benchmark small --seq-length 64 --iters 100
```

Automated CI benchmarks (CPU-verified) are published in [`benchmarks/`](benchmarks/) with timestamps and commit hashes.

## Reporting Security Issues

**Do not report security vulnerabilities in public issues.**

Email: **security@neuramorphic.ai**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We aim to respond within 48 hours and will credit reporters in security advisories (unless you prefer to remain anonymous).

## Security Best Practices

When using NeuraTensor in production:

- **Pin versions:** Use `neuratensor==4.0.2` in requirements.txt, not `neuratensor>=4.0`
- **Verify checksums:** Check SHA256 hashes against release notes
- **Audit dependencies:** Review PyTorch and NumPy versions for known CVEs
- **Sandboxing:** Run untrusted model inference in isolated environments
- **Input validation:** Validate tensor shapes and ranges before model.forward()

## Transparency Commitment

We commit to:

- 🔐 Implementing supply chain security best practices (Trusted Publishing, SLSA)
- 📊 Publishing reproducible benchmark results from CI
- 🐛 Publicly documenting bugs and fixes (see CHANGELOG.md)
- 🚨 Transparent disclosure of security vulnerabilities
- 📝 Honest documentation of limitations and known issues

This is a production-grade SDK, but v4.x is still early. If you find issues, please report them.

---

**Last updated:** 2026-10-29  
**Contact:** support@neuramorphic.ai
