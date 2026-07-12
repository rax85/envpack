# Publishing to PyPI

This document describes the process and commands needed to build and publish new releases of the **envpack** package to PyPI.

---

## 1. Project Information
- **Package Name:** `envpack`
- **PyPI Page:** [https://pypi.org/project/envpack/](https://pypi.org/project/envpack/)
- **Configuration File:** `pyproject.toml`
- **Build Backend:** `setuptools.build_meta`

---

## 2. Prerequisites
Ensure you have the required Python packaging tools installed:

```bash
python3 -m pip install --upgrade build twine
```

---

## 3. Step-by-Step Release Process

### Step 1: Update the Version Number
Before creating a new release, open `pyproject.toml` and update the version string under the `[project]` section:
```toml
[project]
name = "envpack"
version = "0.0.2"  # Update this to the new version (e.g., 0.0.2, 0.1.0)
```

### Step 2: Clean Old Builds
To prevent uploading outdated builds, clean the `dist/` and `build/` directories:
```bash
rm -rf build/ dist/ envpack.egg-info/
```

### Step 3: Build the Distribution Archives
Run the build command to generate the Source Archive (`.tar.gz`) and Built Distribution (`.whl`) files:
```bash
python3 -m build
```

This will create a new set of files inside the `dist/` directory.

### Step 4: Verify the Builds
Verify that your package description and metadata are valid:
```bash
python3 -m twine check dist/*
```
Ensure all checks show `PASSED` before proceeding.

### Step 5: Upload to PyPI
To upload the package, run:
```bash
python3 -m twine upload dist/*
```

When prompted:
- **Username:** Enter `__token__`
- **Password:** Enter your PyPI API token (including the `pypi-` prefix).

#### Optional: Automating Credentials
Instead of entering the token manually every time, you can:
- **Environment Variable:** Set the environment variable before running twine:
  ```bash
  export TWINE_USERNAME="__token__"
  export TWINE_PASSWORD="your-pypi-api-token"
  python3 -m twine upload dist/*
  ```
- **Configuration File (`~/.pypirc`):** Create a `.pypirc` file in your home directory (`~/.pypirc`) with the following contents:
  ```ini
  [distutils]
  index-servers =
      pypi

  [pypi]
  username = __token__
  password = pypi-YOUR_API_TOKEN_HERE
  ```
