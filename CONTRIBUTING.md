# Contributing to GANDALF KRMHD

Thank you for your interest in contributing to GANDALF! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Install dependencies including dev tools
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Verify installation
python -c 'import jax; print(jax.devices())'
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_diagnostics.py

# Run with verbose output
uv run pytest -xvs

# Exclude slow tests
uv run pytest -m "not slow and not benchmark"

# Run only fast tests
uv run pytest -m fast
```

## Code Quality

### Formatting

We use `ruff` for code formatting:

```bash
# Format all code
uv run ruff format

# Check formatting without making changes
uv run ruff format --check
```

### Linting

```bash
# Run linter
uv run ruff check

# Auto-fix issues where possible
uv run ruff check --fix
```

### Type Checking

```bash
# Run type checker
uv run mypy src/krmhd
```

### Pre-commit Checklist

Before committing, ensure:
1. All tests pass: `uv run pytest -m "not slow and not benchmark"`
2. Code is formatted: `uv run ruff format`
3. No linting errors: `uv run ruff check`
4. Type checking passes: `uv run mypy src/krmhd`

## Making Changes

### Branch Naming

- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

### Commit Messages

Follow these conventions:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issue numbers when applicable

Examples:
```
Add kinetic FDT validation tests (Issue #27)
Fix energy conservation in Poisson bracket (Issue #44)
Update documentation for forced turbulence parameters
```

### Pull Request Process

1. Create a new branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Update documentation as needed
5. Ensure all tests pass and code quality checks succeed
6. Push to your fork and submit a pull request
7. Wait for review and address feedback

## Publishing Releases (Maintainers Only)

### PyPI Publishing

The project uses automated PyPI publishing via GitHub Actions. Releases are triggered by pushing version tags.

#### Release Process

1. **Update version** in `pyproject.toml`:
   ```toml
   [project]
   name = "gandalf-krmhd"
   version = "0.2.0"  # Update this
   ```

2. **Commit and tag** the release:
   ```bash
   git add pyproject.toml
   git commit -m "Release v0.2.0"
   git tag v0.2.0
   git push origin main
   git push origin v0.2.0
   ```

3. **Automated workflow** triggers:
   - Builds the package using `uv build`
   - Publishes to PyPI using trusted publishing
   - Creates GitHub Release with artifacts

4. **Verify** the release:
   - Check PyPI: https://pypi.org/project/gandalf-krmhd/
   - Test installation: `pip install gandalf-krmhd==0.2.0`
   - Verify GitHub Release created

#### PyPI Trusted Publishing Setup

The workflow uses PyPI's trusted publishing (no API tokens needed). To set this up:

1. Go to PyPI project settings: https://pypi.org/manage/project/gandalf-krmhd/
2. Navigate to "Publishing" section
3. Add GitHub Actions publisher:
   - Owner: `anjor`
   - Repository: `gandalf`
   - Workflow name: `publish-pypi.yml`
   - Environment: `pypi`

**Note:** Trusted publishing requires the PyPI project to exist first. See "First-Time Release" below.

#### First-Time Release (Manual)

For the very first release (v0.1.0), PyPI requires manual upload since the project doesn't exist yet:

1. **Build package locally**:
   ```bash
   uv build
   # Creates dist/gandalf_krmhd-0.1.0.tar.gz and dist/gandalf_krmhd-0.1.0-py3-none-any.whl
   ```

2. **Create PyPI account** (if needed):
   - Register at https://pypi.org/account/register/
   - Verify your email address

3. **Upload to PyPI**:
   ```bash
   uv publish
   # You'll be prompted for PyPI username and password
   # Or use: uv publish --token <your-api-token>
   ```

4. **Verify upload**:
   ```bash
   # Test in a fresh environment
   python -m venv test_env
   source test_env/bin/activate
   pip install gandalf-krmhd
   python -c "import krmhd; print('Success!')"
   deactivate
   rm -rf test_env
   ```

5. **Configure trusted publishing** (one-time setup):
   - Go to https://pypi.org/manage/project/gandalf-krmhd/settings/publishing/
   - Click "Add a new publisher"
   - Fill in:
     - **PyPI Project Name**: gandalf-krmhd
     - **Owner**: anjor
     - **Repository name**: gandalf
     - **Workflow name**: publish-pypi.yml
     - **Environment name**: pypi
   - Save

6. **Future releases** (v0.1.1+):
   - Simply tag and push: `git tag v0.1.1 && git push origin v0.1.1`
   - GitHub Actions automatically publishes to PyPI (no manual steps!)

#### Manual Publishing (Emergency Only)

If automated workflow fails:

```bash
# Build package
uv build

# Publish (requires PyPI API token)
uv publish --token <your-pypi-token>

# Or use twine
pip install twine
twine upload dist/*
```

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.2.0): New functionality, backward compatible
- **PATCH** version (0.1.1): Bug fixes, backward compatible

For pre-release versions:
- Alpha: `0.1.0a1`
- Beta: `0.1.0b1`
- Release candidate: `0.1.0rc1`

### Release Checklist

Before releasing:

- [ ] All tests pass on main branch
- [ ] Documentation is up to date
- [ ] Release notes prepared (via GitHub Releases)
- [ ] Version number updated in `pyproject.toml`
- [ ] Tag matches version number (e.g., `v0.2.0`)

After releasing:

- [ ] Verify PyPI package: `pip install --upgrade gandalf-krmhd`
- [ ] Test fresh installation in clean environment
- [ ] Announce release (GitHub Discussions, mailing list, etc.)
- [ ] Update conda-forge feedstock (if applicable)

## Docker Container Releases

Docker images are automatically built and published to GitHub Container Registry when tags are pushed.

### Available Images

- `ghcr.io/anjor/gandalf:latest` - Latest release, CPU backend
- `ghcr.io/anjor/gandalf:v0.2.0` - Specific version, CPU backend
- `ghcr.io/anjor/gandalf:latest-metal` - Latest release, Metal backend (macOS)
- `ghcr.io/anjor/gandalf:latest-cuda` - Latest release, CUDA backend (NVIDIA GPUs)

### Testing Docker Images Locally

```bash
# Build CPU image
docker build -t gandalf-krmhd:test -f Dockerfile .

# Build Metal image (macOS)
docker build -t gandalf-krmhd:test-metal -f Dockerfile.metal .

# Build CUDA image (Linux)
docker build -t gandalf-krmhd:test-cuda -f Dockerfile.cuda .

# Test the image
docker run -it gandalf-krmhd:test python -c 'import krmhd; print("Success!")'
```

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/anjor/gandalf/discussions)
- **Bug reports**: Open a [GitHub Issue](https://github.com/anjor/gandalf/issues)
- **Feature requests**: Open a [GitHub Issue](https://github.com/anjor/gandalf/issues) with "enhancement" label
- **Contact**: Email anjor@umd.edu

## Code of Conduct

Be respectful and constructive in all interactions. This is a scientific research project and we welcome contributions from all backgrounds and skill levels.

## License

By contributing to GANDALF, you agree that your contributions will be licensed under the MIT License.
