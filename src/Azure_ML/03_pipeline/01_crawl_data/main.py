"""Main script for step: Crawl Data"""

import os
import argparse
from azureml.core import Run
import crawl_data_questionaire_results_src.main as crawler


print("Crawling data...")


# --- initialization
print("Initialization...")

# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
parser.add_argument("--sql-name-in", type=str, required=True, help="login name for DB")
parser.add_argument("--sql-pw-in", type=str, required=True, help="login PW for DB")
args = parser.parse_args()
output_dir = args.output_dir
sql_name_in = args.sql_name_in
sql_pw_in = args.sql_pw_in

# - get run context
run = Run.get_context()

# - ensure that the output directory exists
print("Ensuring that the output directory exists...")


print("Loading data from Azure SQL DB ...")
crawler.main(sql_name_in, sql_pw_in)

# --- Done
print("Done.")