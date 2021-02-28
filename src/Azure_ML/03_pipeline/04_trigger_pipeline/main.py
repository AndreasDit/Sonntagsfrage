"""Main script for step: Transform Data"""

import argparse
import gzip
import os

import numpy as np
import requests

print("Transforming data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str,
                    required=True, help="input directory")
args = parser.parse_args()
input_dir = args.input_dir

# --- Define pipeline continuation
url_pipeline = "https://sonntagsfrage-etl-pipelines.azurewebsites.net/api/orchestrators/DurableFunctionsHttpStartPostprocess?"
requests.post(url = url_pipeline)

# --- Done
print("Done.")
