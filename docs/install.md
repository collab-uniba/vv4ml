# Installation

## Requirements

Before installing VV4ML, ensure your system meets the following requirements:

* [Python](https://www.python.org/) >= 3.11
* (Optional, but recommended) [uv](https://docs.astral.sh/uv/) >= 0.11

---

## Installing VV4ML

You can install VV4ML using either `pip` or `uv`. We highly recommend using a virtual environment for your Python
projects.

### Option 1: Using `uv` (Preferred)

The preferred and fastest way to install VV4ML is via `uv`. 

If you are managing a project and want to add VV4ML as a dependency to your `pyproject.toml`, run:
```bash
uv add vv4ml
```

Alternatively, if you just want to install it into your active virtual environment, use:
```bash
uv pip install vv4ml
```

### Option 2: Using `pip` (Standard)

The standard way to install VV4ML is via `pip`. We recommend updating `pip` to the latest version before proceeding:
```bash
pip install --upgrade pip
```

Then install the library:
```bash
pip install vv4ml
```