import pyodbc
import yaml
import os
import sys
sys.path.append(os.getcwd())

import src.configs_for_code as cfg
import src.general_utils as general_utils

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)

WORKSHEET_NAME = configs['google']['worksheet_name']
PATH_GOOGLE_SERVICE_ACCOUNT = cfg.PATH_GOOGLE_SERVICE_ACCOUNT

# open connection to Azure SQL DB
# conn, cursor = general_utils.connect_to_azure_sql_db()
conn, cursor = general_utils.connect_to_siteground_sql_db()
