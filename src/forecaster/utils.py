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

    # open connection
    conn_str = 'DRIVER=' + driver + ';SERVER=' + server + ';PORT=1433;DATABASE=' + database + ';UID=' + username + ';PWD=' + password
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

    logger.info("Writing was successful.")


def load_df_from_file(filename, path=PATH_DATAFRAMES):
    """
        Loads a local file into a Pandas Dataframe.

        :param filename: Name of the file that shall be loaded.
        :param path: Path to the file.
        :return: pandas.Dataframe: Returns a Pandas Dataframe created from the loaded file.
    """
    logger.info("Start load_df_from_file() from path " + path + '/' + filename)

    df = pd.read_pickle(f"{path}/{filename}.pkl")

    logger.info("Loading was successful.")
    return df


def set_datecol_as_index_if_needed(df_input):
    """
        This function checks if the dataframe has its date column set as a datetime index. If not it sets the
        column (as defined in the configs) as the datetime index.

        :param df_input: The dataframe which does get checked for datetime index.
        :return: pandas.DataFrame: Returns either the input dataframe unmodified or returns the given dataframe
            with its date column set as an index.
    """
    logger.info("Start set_index_if_needed()")

    ts_idx_exists = 0
    df_tmp = df_input.copy()

    if df_tmp.index.dtype == 'datetime64[ns]':
        ts_idx_exists = 1

    if ts_idx_exists == 0:
        df_tmp = df_tmp.set_index(DATE_COL)

    df_tmp = df_tmp.sort_index()

    df_final = df_tmp
    logger.info("End set_index_if_needed()")
    return df_final

def unset_datecol_as_index_if_needed(df_input):
    """
        This function checks if the dataframe has its date column set as a datetime index. If not it sets the
        column (as defined in the configs) as the datetime index.

        :param df_input: The dataframe which does get checked for datetime index.
        :return: pandas.DataFrame: Returns either the input dataframe unmodified or returns the given dataframe
            with its date column set as an index.
    """
    logger.info("Start unset_datecol_as_index_if_needed()")

    ts_idx_exists = 0
    df_tmp = df_input.copy()

    if df_tmp.index.dtype == 'datetime64[ns]':
        ts_idx_exists = 1

    if ts_idx_exists == 1:
        df_tmp = df_tmp.reset_index(level=DATE_COL)

    df_final = df_tmp
    logger.info("End unset_datecol_as_index_if_needed()")
    return df_final

