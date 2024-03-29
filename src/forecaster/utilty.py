import yaml
import pandas as pd

import utils.logs as logs
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FILE_PATH_LOGGING = configs['logging']['file_path']
PATH_DATAFRAMES = cfg.PATH_DATAFRAMES
DATE_COL = configs['model']['date_col']


def write_df_to_file(df, filename, path=PATH_DATAFRAMES, mode='pkl', force_write=False):
    """
        Saves a Pandas Dataframe to a file.

        :param df: Pandas Dataframe that shall be written to a file.
        :param filename: Name of the file.
        :param path: Path where the file should be written.
    """
    logger.info("Start write_df_to_file() to path " +
                path + '/' + filename + ' with mode ' + mode)

    # Azure has read only file system, hence for deployment this has to be set to true
    RUN_ON_AZURE = configs['general']['run_on_azure']
    if (RUN_ON_AZURE == False) or (force_write==True):
        if mode == 'pkl':
            df.to_pickle(f"{path}/{filename}.pkl", protocol=4)
        if mode == 'parquet':
            df.to_parquet(f"{path}/{filename}.parquet")

    logger.info("Writing was successful.")


def load_df_from_file(filename, path=PATH_DATAFRAMES, mode='pkl'):
    """
        Loads a local file into a Pandas Dataframe.

        :param filename: Name of the file that shall be loaded.
        :param path: Path to the file.
        :return: pandas.Dataframe: Returns a Pandas Dataframe created from the loaded file.
    """
    logger.info("Start load_df_from_file() from path " +
                path + '/' + filename + ' with mode ' + mode)

    if mode == 'pkl':
        df = pd.read_pickle(f"{path}/{filename}.pkl")
    if mode == 'parquet':
        df = pd.read_parquet(f"{path}/{filename}.parquet")

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
