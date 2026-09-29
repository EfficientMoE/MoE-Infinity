from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def _read(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def test_stable_release_is_github_primary_and_pypi_is_non_blocking():
    workflow = _read("publish.yml")

    assert "workflow_dispatch:" in workflow
    assert "MOE_STORE_RELEASE_TAG" in workflow
    assert "EfficientMoE/moe-store.git@${MOE_STORE_TAG}" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "needs: wheel" in workflow
    assert "gh release create" in workflow
    assert "shopt -s nullglob" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "actions/upload-release-asset" not in workflow
    assert "name: Mirror optional sdist to PyPI" in workflow
    assert "needs: release" in workflow
    assert "continue-on-error: true" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow
    assert "id-token: write" in workflow


def test_nightly_build_uploads_artifacts_without_publishing():
    workflow = _read("publish-test.yml")

    assert "pypa/gh-action-pypi-publish" not in workflow
    assert "actions/upload-artifact@" in workflow
    assert "Publish sdist to PyPI" not in workflow


def test_no_separate_pypi_workflow_can_gate_or_drift_from_release():
    assert not (WORKFLOWS / "publish-pypi.yml").exists()


def test_installation_docs_make_github_releases_authoritative():
    readme = (ROOT / "README.md").read_text()
    release = (ROOT / "RELEASE.md").read_text()
    security = (ROOT / "SECURITY.md").read_text()

    assert "GitHub Releases are the authoritative" in readme
    assert "gh release download" in readme
    assert "--pattern 'moe_store-*.whl'" in readme
    assert '--pattern "moe_infinity-*${PYTAG}*manylinux*.whl"' in readme
    assert "`gh` (GitHub CLI)" in readme
    assert "moe-store.git@v" in readme
    assert "PyPI publishing is optional" in release
    assert "does not gate the GitHub Release" in release
    assert "non-blocking job" in release
    assert "Latest GitHub Release" in security
