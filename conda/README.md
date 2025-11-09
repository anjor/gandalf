# Conda-Forge Distribution

This directory contains the recipe template for publishing GANDALF to conda-forge.

## Overview

Conda-forge is a community-driven package distribution system for conda. Publishing to conda-forge makes GANDALF easily installable via:

```bash
conda install -c conda-forge gandalf-krmhd
```

## Prerequisites

Before submitting to conda-forge:

1. **Package must be published to PyPI first**
   - Conda-forge recipes typically wrap PyPI packages
   - See `CONTRIBUTING.md` for PyPI publishing instructions
   - Verify: https://pypi.org/project/gandalf-krmhd/

2. **GitHub repository must be public**
   - Already satisfied: https://github.com/anjor/gandalf

3. **LICENSE file must exist**
   - Ensure LICENSE file is present in repository root
   - Currently using MIT license

## Submission Process

### Step 1: Fork staged-recipes

```bash
# Fork the conda-forge/staged-recipes repository via GitHub UI
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes
```

### Step 2: Create recipe branch

```bash
# Create a new branch for your recipe
git checkout -b gandalf-krmhd
```

### Step 3: Copy recipe

```bash
# Create recipe directory
mkdir recipes/gandalf-krmhd

# Copy the meta.yaml from this directory
cp /path/to/gandalf/conda/meta.yaml recipes/gandalf-krmhd/

# Update version and SHA256
# Get SHA256 from PyPI: https://pypi.org/project/gandalf-krmhd/#files
```

### Step 4: Update meta.yaml

Edit `recipes/gandalf-krmhd/meta.yaml`:

1. **Update version** to match PyPI release:
   ```yaml
   {% set version = "0.1.0" %}  # Match PyPI version
   ```

2. **Add SHA256** from PyPI package:
   - Go to https://pypi.org/project/gandalf-krmhd/#files
   - Click "view hashes" for the `.tar.gz` file
   - Copy SHA256 hash:
     ```yaml
     source:
       url: https://pypi.io/packages/source/g/gandalf-krmhd/gandalf_krmhd-{{ version }}.tar.gz
       sha256: abc123...  # Paste SHA256 here
     ```

3. **Verify dependencies** match `pyproject.toml`

4. **Update recipe-maintainers**:
   ```yaml
   extra:
     recipe-maintainers:
       - anjor  # Your GitHub username
   ```

### Step 5: Test recipe locally (optional but recommended)

```bash
# Install conda-build
conda install conda-build

# Build the recipe
conda build recipes/gandalf-krmhd

# Test installation
conda create -n test-gandalf gandalf-krmhd --use-local
conda activate test-gandalf
python -c "import krmhd; print('Success!')"
conda deactivate
conda env remove -n test-gandalf
```

### Step 6: Submit pull request

```bash
# Commit your recipe
git add recipes/gandalf-krmhd/
git commit -m "Add gandalf-krmhd recipe"

# Push to your fork
git push origin gandalf-krmhd
```

Then:
1. Go to https://github.com/conda-forge/staged-recipes
2. Create a Pull Request from your branch
3. Fill in PR template (auto-generated)
4. Wait for CI checks to pass (automatic)
5. Address any reviewer feedback

### Step 7: Post-submission

Once merged:
1. A new repository is created: `conda-forge/gandalf-krmhd-feedstock`
2. You'll be added as a maintainer automatically
3. Package appears on conda-forge: https://anaconda.org/conda-forge/gandalf-krmhd
4. Users can install with: `conda install -c conda-forge gandalf-krmhd`

## Updating the Package

After initial submission, updates are managed via the feedstock repository:

### Automatic Updates (Preferred)

Conda-forge bots automatically detect new PyPI releases and create PRs:

1. Publish new version to PyPI (see `CONTRIBUTING.md`)
2. Wait ~1-2 hours for bot to detect update
3. Bot creates PR in `conda-forge/gandalf-krmhd-feedstock`
4. Review and merge the PR
5. New conda package builds automatically

### Manual Updates

If automatic updates fail or dependencies change:

```bash
# Clone the feedstock
git clone https://github.com/conda-forge/gandalf-krmhd-feedstock.git
cd gandalf-krmhd-feedstock

# Create update branch
git checkout -b update-v0.2.0

# Edit recipe/meta.yaml
# - Update version
# - Update SHA256
# - Update dependencies if needed

# Push and create PR
git commit -am "Update to v0.2.0"
git push origin update-v0.2.0
```

## Troubleshooting

### Issue: "Package not found on PyPI"

**Solution**: Ensure package is published to PyPI first. Conda-forge pulls from PyPI.

### Issue: "Build backend uv_build not supported"

**Not an issue**: The conda recipe uses `pip install` which can install packages built with any PEP 517 backend (including uv_build). The PyPI package is built with uv_build, but conda-forge installs it via pip with setuptools as a build dependency.

### Issue: "SHA256 mismatch"

**Solution**:
1. Download package from PyPI
2. Compute SHA256: `openssl sha256 gandalf_krmhd-0.1.0.tar.gz`
3. Update meta.yaml with correct hash

### Issue: "Import test fails"

**Solution**: Check that all dependencies are available in conda-forge:
- Most Python packages are available
- JAX is available as `jax`
- If dependency missing, it needs its own conda-forge recipe first

### Issue: "Build fails"

**Solution**: Common causes:
- Missing build dependencies in `host` section
- Missing runtime dependencies in `run` section
- Platform-specific issues (check CI logs)

## Platform Support

### Current Configuration

The recipe uses `noarch: python`, meaning:
- ✅ Works on: Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows
- ✅ Pure Python package (no compiled extensions)
- ⚠️ GPU support depends on JAX installation (handled separately)

### GPU Support

Users can install GPU-enabled JAX separately:

```bash
# Install GANDALF
conda install -c conda-forge gandalf-krmhd

# Then install JAX with GPU support
# For CUDA (Linux)
conda install -c conda-forge jax cuda-nvcc

# For Metal (macOS), pip is required
pip install jax-metal
```

**Note**: JAX Metal (Apple Silicon) is not available on conda-forge, users must use pip.

## Resources

- **Conda-forge documentation**: https://conda-forge.org/docs/
- **Staged-recipes repo**: https://github.com/conda-forge/staged-recipes
- **Recipe format**: https://conda-forge.org/docs/maintainer/adding_pkgs.html
- **Feedstock maintenance**: https://conda-forge.org/docs/maintainer/updating_pkgs.html
- **Package search**: https://anaconda.org/conda-forge

## Maintainer Notes

After becoming a conda-forge maintainer:

1. **Watch feedstock repo**: Get notifications for new issues/PRs
2. **Review bot PRs**: Automatic updates need approval
3. **Respond to issues**: Help users with installation problems
4. **Keep dependencies updated**: Monitor for dependency deprecations
5. **Test before merging**: Especially for major version updates

## Questions?

- **Conda-forge Gitter**: https://gitter.im/conda-forge/conda-forge.github.io
- **Staged-recipes issues**: https://github.com/conda-forge/staged-recipes/issues
- **GANDALF issues**: https://github.com/anjor/gandalf/issues
