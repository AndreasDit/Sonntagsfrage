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

import pandas as pd
import yaml

sys.path.append(os.getcwd())
sys.path.append('./src/azure_functions/')
import utils.logs as logs
import forecaster.utilty as utils
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)


# get workspace and datastore
env = Env()
env.read_env("src/azure_functions/Azure_ML/foundation.env")
workspace = Workspace(env("AZURE_SUBSCRIPTION_ID"), env("RESOURCE_GROUP"), env("WORKSPACE_NAME"))
# datastore = Datastore.get(workspace, env("SOME_EXTERNAL_ADLS_GEN2_DATASTORE_NAME"))

# sql_datastore_name ="azuresqldatastore"
# server_nam e =os.getenv("SQL_SERVERNAME", "<my_server_name>") # Name of the Azure SQL server
# database_nam e =os.getenv("SQL_DATABASENAME", "<my_database_name>") # Name of the Azure SQL database
# usernam e =os.getenv("SQL_USER_NAME", "<my_sql_user_name>") # The username of the database user.
# passwor d =os.getenv("SQL_USER_PASSWORD", "<my_sql_user_password>") # The password of the database user.
#
# sql_datastore = Datastore.register_azure_sql_database(
#     workspace=ws,
#     datastore_name=sql_datastore_name,
#     server_name=server_name,  # name should not contain fully qualified domain endpoint
#     database_name=database_name,
#     username=username,
#     password=password,
#     endpoint='database.windows.net')

# # create a FileDataset pointing to the existing files in folder mnist-fashion on the data lake account
# datastore_paths = [(datastore, "mnist-fashion")]
# dataset = Dataset.File.from_files(path=datastore_paths)
#
# # register the dataset to enable later reuse
# dataset.register(workspace, "mnist-fashion", "Sample dataset", create_new_version=True)
#
# # get the dataset (for the case we have no dataset instance at this point)
# dataset = Dataset.get_by_name(workspace=workspace, name="mnist-fashion")
#
# # download the files to local folder
# # note: this is only one of multiple options. for instance, on Unix machines, the dataset can also be mounted
# directory_of_this_script = os.path.dirname(os.path.realpath(__file__))
# dataset.download(directory_of_this_script)

