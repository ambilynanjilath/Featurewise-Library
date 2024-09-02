# FeatureWise

[![PyPI version](https://badge.fury.io/py/featurewise.svg)](https://badge.fury.io/py/featurewise)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

FeatureWise is a Python package for feature engineering that provides a set of tools for data transformation, imputation, encoding, scaling, and feature creation. This package comes with an interactive Streamlit interface that allows users to easily apply these transformations to their datasets.

## Features

- Create polynomial features
- Handle and extract date-time features
- Encode categorical data using various encoding techniques
- Impute missing values with different strategies
- Normalize and scale data using multiple scaling methods
- Interactive Streamlit interface for easy usage

## Installation

You can install FeatureWise from PyPI using `pip`:

```bash
pip install FeatureWise==1.0.0
```
## Quick Start
After installing the package, run the FeatureWise interface using:

```bash
run featurewise
```
This will open a Streamlit app where you can upload your dataset and start applying transformations.

## Usage
### Command-Line Interface
To launch the Streamlit app, simply use the command:
```bash
run featurewise
```
### Importing Modules in Python
You can also use FeatureWise modules directly in your Python scripts:
```bash
from featurewise.imputation import MissingValueImputation
from featurewise.encoding import FeatureEncoding
from featurewise.imputation import MissingValueImputation
from featurewise.encoding import FeatureEncoding
from featurewise.scaling import DataNormalize
from featurewise.date_time_features import DateTimeExtractor
from featurewise.create_features import PolynomialFeaturesTransformer
```

## Modules Overview

The `featurewise` package provides several modules for different data transformation tasks:

- **create_features.py** - Generate polynomial features.
- **date_time_features.py** - Extract and handle date-time related features.
- **encoding.py** - Encode categorical features using techniques like Label Encoding and One-Hot Encoding.
- **imputation.py** - Handle missing values with multiple imputation strategies.
- **scaling.py** - Normalize and scale numerical features.

Each of these modules is described in detail below.

### 1. `create_features.py`
<Details to be provided later>

### 2. `date_time_features.py`
<Details to be provided later>

### 3. `encoding.py`
<Details to be provided later>

### 4. `imputation.py`
<Details to be provided later>

### 5. `scaling.py`
<Details to be provided later>

## Requirements

Before installing, please make sure you have the following packages installed:

- Python >= 3.7
- Streamlit
- Pandas
- NumPy
- scikit-learn
- st-aggrid

For more detailed information, see the `requirements.txt` file.

## Contributing

We welcome contributions! Please read our Contributing Guidelines for more details.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

Special thanks to all the libraries and frameworks that have helped in developing this package, including:

- [Streamlit](https://www.streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)


