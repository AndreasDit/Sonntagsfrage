"""Main script for step: Transform Data"""

import argparse
import gzip
import os

import numpy as np
import requests

print("Transforming data...")


# --- initialization
print("Initialization...")

# --- Define pipeline continuation
url_pipeline = "https://sonntagsfrage-etl-pipelines.azurewebsites.net/api/orchestrators/test?"
requests.post(url = url_pipeline)

# --- Done
print("Done.")
