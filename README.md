# Auditing Synthetic Data Generation

This repo contains the code of a paper on auditing the statistics used by synthetic data generators.

## Running the code

### Installation

First, install `poetry` and install the package. This requires Python >= 3.9.

```
poetry install
```

### Loading the data

Download the [1% Census Teaching file](https://www.ons.gov.uk/census/2011census/2011censusdata/censusmicrodata/microdatateachingfile), and save it as `main.csv`.

Once that is done, open the Python interpreter (`poetry run python`), and type the following to split the dataset between training and testing.

```
from utils import split_data

split_data()
```

### Generating synthetic data

The non-decomposable synthetic datasets used for the utility analysis are pre-computed. For this, run jupyter (`poetry run jupyter`) and, in the browser, open `Generating all non-tailored datasets.ipynb`. Run all cells of this notebook.

### Replicating our analysis

The notebooks `Use Case X.ipynb` contain the utility analysis for each application (Section 4.3). The `Auditing USE CASE X.ipynb` notebooks contain the auditing analysis for each application (Section 4.4).