"""
Register a FileDataset and download the contained files.
Notes: - Besides the FileDataset, there is also TabularDatasets which are better suited for tabular data.
         See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets for more details.
       - Besides downloading, datasets can also be mounted into the file system (Unix). Depending on the data volumes,
         a mount might be the better option.
"""

import os
import sys
import azureml as ml
from azureml.core import Workspace, Datastore, Dataset
from environs import Env
import numpy as np

import pandas as pd
import yaml

sys.path.append(os.getcwd())
sys.path.append('./src/azure_functions/')
import forecaster.prepare_data as prep
import forecaster.feat_engineering as feat
import forecaster.utilty as utils
import utils.logs as logs
import forecaster.utilty as utils
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

def main():
  # get workspace and datastore
  env = Env()
  env.read_env("Azure_ML/foundation.env")
  ws = Workspace(env("AZURE_SUBSCRIPTION_ID"), env("RESOURCE_GROUP"), env("WORKSPACE_NAME"))
  # datastore = Datastore.get(ws, env("SOME_EXTERNAL_BLOB_DATASTORE_NAME"))
  datastore = Datastore.get_default(ws)

  df_all_data = prep.load_data()

  df_with_features = feat.generate_features(df_all_data)
 
  Dataset.Tabular.register_pandas_dataframe(df_with_features, (datastore, 'azure-ml-datasets'), 'survey_data_with_all_features')

main()




