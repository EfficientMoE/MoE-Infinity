# Package Release Guide

This document describes the process of releasing a new version of the MoE-Infinity package.

## Automated Release Process

Stable releases are automated through GitHub Actions workflows in `.github/workflows/`:

- `.github/workflows/publish.yml`: builds and publishes tagged stable releases (`v*`) to PyPI and creates a GitHub release.
- `.github/workflows/publish-test.yml`: publishes nightly pre-release builds from `main` to PyPI.
- `.github/workflows/build-test.yml`: build validation for pull requests.

### Steps to Release a New Version
To release a new version, such as version 1.0.0, please adhere to the following procedure:

1. Update Version:
   - Update `moe_infinity/__init__.py` (`__version__ = "..."`) to the new stable version.
   - Ensure `setup.py` remains `version=os.getenv("MOEINF_VERSION", "...")` and update the default fallback version there to match the new stable version.
   - If needed, bump `NIGHTLY_BASE_VERSION` in `.github/workflows/publish-test.yml` to the next planned stable series so nightly dev builds sort correctly.
2. Commit Changes: Commit these changes with an appropriate commit message that summarizes the update, such as "Update version for 1.0.0 release".
3. Create and Push Tag: Tag the latest commit with the new version number and push the tag to the repository. Use the following commands to accomplish this:
    ```bash
    git tag v1.0.0
    git push origin v1.0.0
    ```

Upon a successful tag push, the release workflow will create a release draft, build artifacts, and publish the package to PyPI.


## Manual Package Building and Publishing

For developers who prefer to manually build and publish their package to PyPI, the following steps provide a detailed guide to execute this process effectively.

1. Start by cloning the repository and navigating to the root directory of the package:
    ```bash
    git clone https://github.com/EfficientMoE/MoE-Infinity.git
    cd MoE-Infinity
    ```
2. Install the required dependencies to build the package:
    ```bash
    pip install -r requirements.txt
    pip install build
    ```
3. Build the source distribution and wheel for the package using:
    ```bash
    python -m build
    ```
    This command generates the package files in the `dist/` directory.
4. Upload the built package to the PyPI repository using `twine`:
    ```bash
    twine upload dist/*
    ```
    Ensure that you have the necessary credentials configured for `twine` to authenticate to PyPI.


To build the package wheel for multiple Python versions, you should execute the build process individually for each version by specifying the corresponding Python interpreter.
