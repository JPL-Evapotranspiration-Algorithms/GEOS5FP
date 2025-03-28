# `GEOS5FP` Python Package

[![CI](https://github.com/JPL-Evapotranspiration-Algorithms/geos5fp/actions/workflows/ci.yml/badge.svg)](https://github.com/JPL-Evapotranspiration-Algorithms/geos5fp/actions/workflows/ci.yml)

The `geos5fp` Python package generates rasters of near-real-time GEOS-5 FP near-surface meteorology.

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
NASA Jet Propulsion Laboratory 329G

## Prerequisites

This package calls the `wget` command, which must be installed on your system.

For macOS, the `wget` command can be installed with Homebrew:

```
brew install wget
```

The `wget` command can also be installed with `mamba`:

```
mamba install wget
```

## Installation

This package is available on PyPi as a [pip package](https://pypi.org/project/geos5fp/) called `geos5fp`.

```bash
pip install geos5fp
```

## Usage

Import this package as `geos5fp`.

```python
import geos5fp
```
