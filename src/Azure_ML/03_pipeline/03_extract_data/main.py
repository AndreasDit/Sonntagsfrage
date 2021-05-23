"""Main script for step: Extract Data"""

import argparse
import os
import urllib.request
import forecaster.prepare_data as prep
import forecaster.utilty as utils

from azureml.core import Run

print("Extracting data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str, required=True, help="input directory")
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir

# - get run context
run = Run.get_context()
# - ensure that the output directory exists
print("Ensuring that the output directory exists...")
os.makedirs(output_dir, exist_ok=True)


# # --- download and save data to output directory
# # note: we also could have used AzureML's dataset feature here and register/version the dataset
# print("Downloading and saving data...")
# files_to_download = {
#     "t10k-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k"
#                                  "-images-idx3-ubyte.gz",
#     "t10k-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/t10k"
#                                  "-labels-idx1-ubyte.gz",
#     "train-images-idx3-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train"
#                                   "-images-idx3-ubyte.gz",
#     "train-labels-idx1-ubyte.gz": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train"
#                                   "-labels-idx1-ubyte.gz",
# }

# for file_to_download_name in files_to_download:
#     print(f"Downloading file '{file_to_download_name}...")
#     file_to_download_path = os.path.join(output_dir, file_to_download_name)
#     file_to_download_url = files_to_download[file_to_download_name]
#     urllib.request.urlretrieve(file_to_download_url, file_to_download_path)

print("Loading data from Azure SQL DB ...")
df_all_data = prep.load_data()
output_path = os.path.join(output_dir)

output_fname = 'df_all_data'
mode = 'parquet'
print("Writing file "+ output_fname +"."+mode+" to path "+output_path+" ...")
utils.write_df_to_file(df_all_data, output_fname, output_path, mode, force_write=True)

# --- Done
print("Done.")