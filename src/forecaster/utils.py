import pyodbc
import yaml
import pandas as pd

import src.forecaster.logs as logs
import src.forecaster.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FILE_PATH_LOGGING = configs['logging']['file_path']
PATH_DATAFRAMES = cfg.PATH_DATAFRAMES

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

    # open connection
    conn_str = 'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password
    logger.info('Connection string is as follows: ' + conn_str)
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    return conn, cursor


def write_df_to_file(df, filename, path=PATH_DATAFRAMES):
    """
        Saves a Pandas Dataframe to a file.

        :param df: Pandas Dataframe that shall be written to a file.
        :param filename: Name of the file.
        :param path: Path where the file should be written.
    """
    logger.info("Start write_df_to_file() to path " + path + '/' + filename)

    df.to_pickle(f"{path}/{filename}.pkl", protocol=4)


def load_df_from_file(filename, path=PATH_DATAFRAMES):
    """
        Loads a local file into a Pandas Dataframe.

        :param filename: Name of the file that shall be loaded.
        :param path: Path to the file.
        :return: pandas.Dataframe: Returns a Pandas Dataframe created from the loaded file.
    """
    logger.info("Start load_df_from_file() from path " + path + '/' + filename)

    df = pd.read_pickle(f"{path}/{filename}.pkl")

    return df





