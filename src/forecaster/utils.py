import pyodbc
import yaml

import logs
import configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FILE_PATH_LOGGING = configs['logging']['file_path']

def connect_to_azure_sql_db():
    """
        This function creates a connection to a Azure SQL DB where.

        :return: pandas.DataFrame: Returns one dataframe with the combinated data from all input sources.
    """

    # import variables

    # set defaults for azure sql datbse
    server = 'sonntagsfrage-server.database.windows.net'
    database = 'sonntagsfrage-sql-db'
    username = configs['azure']['sql_db_name']
    password = configs['azure']['sql_db_pw']
    driver = '{ODBC Driver 17 for SQL Server}'

    # open connection
    conn_str = 'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password
    logging.info(conn_str)
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
