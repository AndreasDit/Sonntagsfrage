"""Main script for step: Transform Data"""

import argparse
import gzip
import os

import numpy as np
import requests
from google_spreadsheets import load_data_spreadsheet as google


print("Transforming data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str,
                    required=True, help="input directory")
args = parser.parse_args()
input_dir = args.input_dir

# --- Load results to google spreadsheets
google.main()

# --- Done
print("Done.")
