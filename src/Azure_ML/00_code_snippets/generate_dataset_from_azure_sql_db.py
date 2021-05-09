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
  print(datastore)

  # import variables
  sql_name = os.getenv('sql_db_name')
  sql_pw = os.getenv('sql_db_pw')

  # set defaults for azure sql datbse
  server = 'sonntagsfrage-server.database.windows.net'
  database = 'sonntagsfrage-sql-db'
  username = sql_name
  password = sql_pw 
  driver = '{ODBC Driver 17 for SQL Server}'

  server_name = configs['azure']['server_name']
  database = configs['azure']['database']
  username = configs['azure']['sql_db_name']
  password = configs['azure']['sql_db_pw']
  driver = configs['azure']['driver']
  port = configs['azure']['port']
  s_port = str(port)
  
  sql_datastore_name ="sonntagsfrageazuresqldatastore"
  # server_name =os.getenv("SQL_SERVERNAME", "<my_server_name>") # Name of the Azure SQL server
  # database_name =os.getenv("SQL_DATABASENAME", "<my_database_name>") # Name of the Azure SQL database
  # username =os.getenv("SQL_USER_NAME", "<my_sql_user_name>") # The username of the database user.
  # password =os.getenv("SQL_USER_PASSWORD", "<my_sql_user_password>") # The password of the database user.

  # server_name = 'sonntagsfrageserver'

  # sonntagsfrage-server.database.windows.net
  sql_datastore = Datastore.register_azure_sql_database(
      workspace=ws,
      datastore_name=sql_datastore_name,
      server_name=server_name,  # name should not contain fully qualified domain endpoint
      database_name=database,
      username=username,
      password=password,
      endpoint='database.windows.net')
  
  # sql_dataset = Dataset.Tabular.from_sql_query((sql_datastore, 'SELECT * FROM my_table'))

  # # create a FileDataset pointing to the existing files in folder mnist-fashion on the data lake account
  # datastore_paths = [(datastore, "mnist-fashion")]
  # dataset = Dataset.File.from_files(path=datastore_paths)

  # register the dataset to enable later reuse
  # dataset.register(ws, "mnist-fashion", "Sample dataset", create_new_version=True)

  df_all_data = prep.load_data()

  df_with_features = feat.generate_features(df_all_data)
 
  Dataset.Tabular.register_pandas_dataframe(df_with_features, (datastore, 'azure-ml-datasets'), 'survey_data_with_all_features')

#
# # get the dataset (for the case we have no dataset instance at this point)
# dataset = Dataset.get_by_name(workspace=workspace, name="mnist-fashion")
#
# # download the files to local folder
# # note: this is only one of multiple options. for instance, on Unix machines, the dataset can also be mounted
# directory_of_this_script = os.path.dirname(os.path.realpath(__file__))
# dataset.download(directory_of_this_script)


main()




