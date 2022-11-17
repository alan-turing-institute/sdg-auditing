"""Various utilities for the RSS utility analysis.

This file contains code to generate, load, and save synthetic datasets,
according to different desiderata. The goal is to reduce overlap between
different notebooks and scripts.

"""

import json
import numpy as np
import os
import pandas as pd
import tqdm
import tqdm.notebook

# The key library to generate synthetic datasets.
from reprosyn.methods import MST, CTGAN, PATEGAN, PRIVBAYES, DS_PRIVBAYES, SYNTHPOP

from sklearn.model_selection import train_test_split

import sys
sys.path.append('..')
from raw_ipf import IPF

from mbi import Domain, Dataset, FactoredInference


with open("metadata.json", "r") as ff:
    metadata = json.loads(ff.read())


# Only called once, but left here for reference.
def split_data():
    df = pd.read_csv("main.csv").drop(columns=["Person ID"]).astype("str")
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=1)
    df_train.to_csv("main_train.csv", index=False)
    df_test.to_csv("main_test.csv", index=False)

def load_data(train=True, test=False):
    if train and test:
        return pd.read_csv("main.csv").drop(columns=["Person ID"]).astype("str")
    else:
        filename = f"main_{'train' if train else 'test'}.csv"
        return pd.read_csv(filename).astype("str")


## Part 1: generating datasets agnostic to use cases.
target_folder = "generated_datasets"
if not os.path.exists(target_folder):
    os.mkdir(target_folder)


def run_generator(df, method_name, seed, generator_class, output_size, kwargs):
    # Check whether the destination already exists.
    destination = f"{target_folder}/{method_name}/dataset_{seed}.csv"
    if os.path.exists(destination):
        return
    # Otherwise, generate the data.
    np.random.seed(seed)
    generator = generator_class(dataset=df, metadata=metadata, size=output_size, **kwargs)
    generator.run()
    dataset = generator.output
    # Save the dataset to disk.
    dataset.to_csv(destination, index=False)


def load_synthetic_datasets(methods):
    synthetic_datasets = {}
    for method_name in methods:
        synthetic_datasets[method_name] = L = []
        root = f"{target_folder}/{method_name}"
        for dataset_name in os.listdir(root):
            L.append(pd.read_csv(f"{root}/{dataset_name}"))
    return synthetic_datasets


## Part 2: Tailored methods.


def custom_ipf(df, columns, folder, seed, marginals, output_size):
    # Check whether the destination exists (copy-pasted).
    destination = f"{target_folder}/{folder}/dataset_{seed}.csv"
    if not os.path.exists(f"{target_folder}/{folder}"):
        os.mkdir(f"{target_folder}/{folder}")
    if os.path.exists(destination):
        return
    # Restrict the metadata 
    metadata_map = {m['name']: m for m in metadata}
    metadata_columns = [metadata_map[c] for c in columns]
    # Use IPF on the restricted set of columns.
    np.random.random(seed)
    ipf = IPF(
        dataset=df[columns],
        metadata=metadata_columns,
        size=output_size,
        marginals=marginals,
        iter_tolerance=1e-3,
    )
    ipf.run()
    ipf.output.to_csv(destination, index=False)


def custom_mst(df, columns, folder, seed, marginals, output_size, sigma=0.001):
    # Check whether the destination exists (copy-pasted).
    destination = f"{target_folder}/{folder}/dataset_{seed}.csv"
    if not os.path.exists(f"{target_folder}/{folder}"):
        os.mkdir(f"{target_folder}/{folder}")
    if os.path.exists(destination):
        return
    # We have a custom method here.
    encoder = {
        c["name"]: {v: i for i, v in enumerate(c["representation"])} for c in metadata
    }
    decoder = {
        column_name: {i: v for v, i in enc.items()}
        for column_name, enc in encoder.items()
    }
    # Encoding the target dataset.
    dataset_encoded = df[columns].copy()
    for c in columns:
        dataset_encoded[c] = dataset_encoded[c].replace(encoder[c])

    # Extract the domain from the metadata.
    domain = Domain(columns, [len(encoder[c]) for c in columns])
    dataset_mbi = Dataset(dataset_encoded, domain)

    # Get measurements for all 2-way histogram.
    marginals = [list(columns[c] for c in m) for m in marginals]
    histograms = [dataset_mbi.project(m).datavector() for m in marginals]
    measurements = [
        (np.eye(h.size), h, sigma, m) for m, h in zip(marginals, histograms)
    ]

    # Use the MBI procedure for the rest.
    engine = FactoredInference(domain, log=False)
    model = engine.estimate(measurements, engine="MD")
    synth = model.synthetic_data(rows=output_size)

    # Decoding the dataset.
    dataset_decoded = synth.df.copy()
    for c in columns:
        dataset_decoded[c] = dataset_decoded[c].replace(decoder[c])
    # Save to destination.
    dataset_decoded.to_csv(destination, index=False)
