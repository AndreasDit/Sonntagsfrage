import pyodbc
import yaml
import mysql
import mysql.connector

import src.logs as logs
import src.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FILE_PATH_LOGGING = configs['logging']['file_path']
PATH_DATAFRAMES = cfg.PATH_DATAFRAMES
DATE_COL = configs['model']['date_col']


def connect_to_azure_sql_db():
    """
        This function creates a connection to the underlying Azure SQL DB with the data.

        :return: connection: Returns the connection to the DB.
        :return: cursor: Returns the cursor which is used to perform database operations on the Azure SQL DB.
    """
    logger.info('Start connect_to_azure_sql_db()')

    # set defaults for azure sql datbse
    server = configs['azure']['server']
    database = configs['azure']['database']
    username = configs['azure']['sql_db_name']
    password = configs['azure']['sql_db_pw']
    driver = configs['azure']['driver']
    port = configs['azure']['port']
    s_port = str(port)

    # open connection
    conn_str = 'DRIVER=' + driver + ';SERVER=' + server + ';PORT=' + s_port + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    return conn, cursor


def connect_to_siteground_sql_db():
    """
        This function creates a connection to the underlying Azure SQL DB with the data.

        :return: connection: Returns the connection to the DB.
        :return: cursor: Returns the cursor which is used to perform database operations on the Azure SQL DB.
    """
    logger.info('Start connect_to_azure_sql_db()')

    # set defaults for azure sql datbse
    server = configs['siteground']['server']
    database = configs['siteground']['database']
    username = configs['siteground']['sql_db_name']
    password = configs['siteground']['sql_db_pw']

    # open connection
    conn = mysql.connector.connect(user=username, password=password,
                                   host=server,
                                   database=database)
    cursor = conn.cursor()

    return conn, cursor
