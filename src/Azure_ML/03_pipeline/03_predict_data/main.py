"""Main script for step: Transform Data"""

import argparse
import gzip
import os

import numpy as np
from azureml.core import Run, Dataset, Datastore
import forecaster.feat_engineering as feat
import forecaster.utilty as utils
import forecaster.model as model
import forecaster.output as output

print("Transforming data...")


# --- initialization
print("Initialization...")
# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str,
                    required=True, help="input directory")
parser.add_argument("--output-dir", type=str,
                    required=True, help="output directory")
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
# - get run context
run = Run.get_context()
# - ensure that the output directory exists
print("Ensuring that the output directory exists...")
os.makedirs(output_dir, exist_ok=True)


# # --- load data
# def load_gz_data(path):
#     # train labels
#     with gzip.open(os.path.join(path, "train-labels-idx1-ubyte.gz"), "rb") as label_path:
#         train_labels = np.frombuffer(label_path.read(), dtype=np.uint8, offset=8)

#     # train images
#     with gzip.open(os.path.join(path, "train-images-idx3-ubyte.gz"), "rb") as image_path:
#         train_images = np.frombuffer(image_path.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)

#     # test labels
#     with gzip.open(os.path.join(path, "t10k-labels-idx1-ubyte.gz"), "rb") as label_path:
#         test_labels = np.frombuffer(label_path.read(), dtype=np.uint8, offset=8)

#     # test images
#     with gzip.open(os.path.join(path, "t10k-images-idx3-ubyte.gz"), "rb") as image_path:
#         test_images = np.frombuffer(image_path.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 28, 28)

#     return train_images, train_labels, test_images, test_labels


# train_images, train_labels, test_images, test_labels = load_gz_data(input_dir)


# # --- convert label files to .npz format (for consistency)
# print(f"Converting label files to .npz format for consistency...")
# np.savez_compressed(os.path.join(output_dir, "train-labels-transformed.npz"), train_labels)
# np.savez_compressed(os.path.join(output_dir, "t10k-labels-transformed.npz"), test_labels)


# # --- normalize and reshape the images
# print(f"Reshaping and normalizing images...")
# train_images = train_images / 255.0
# np.savez_compressed(os.path.join(output_dir, "train-images-transformed.npz"), train_images)

# test_images = test_images / 255.0
# np.savez_compressed(os.path.join(output_dir, "t10k-images-transformed.npz"), test_images)

# --- get full paths
input_path = os.path.join(input_dir)
output_path = os.path.join(output_dir)

# --- load input
print(f"Load file from last step ...")
df_with_features = utils.load_df_from_file(
    'df_with_features', input_path, 'parquet')

# --- calc predictions
print(f"Add features to survey data ...")
df_with_preds, df_metrics = model.combined_restults_from_all_algorithms(
    df_with_features)

# --- define output parameters
output_fname = 'df_with_preds'
mode = 'parquet'

# --- get ws from run
run = Run.get_context()
ws = run.experiment.workspace
datastore = Datastore.get_default(ws)

# --- register preds
df_for_register = utils.unset_datecol_as_index_if_needed(df_with_preds)
Dataset.Tabular.register_pandas_dataframe(df_for_register, (datastore, 'azure-ml-datasets'), 'sonntagsfrage_preds')

# --- register metrics
df_for_register = utils.unset_datecol_as_index_if_needed(df_metrics)
Dataset.Tabular.register_pandas_dataframe(df_for_register, (datastore, 'azure-ml-datasets'), 'sonntagsfrage_metrics')

# --- write output to Azure SQL DB
print("Writing file to Azure SQL DB ...")
output.export_results(df_with_preds)

# --- write output to file
print("Writing file " + output_fname + "."+mode+" to path "+output_path+" ...")
utils.write_df_to_file(df_with_preds, output_fname, output_path, mode)

# --- Done
print("Done.")
