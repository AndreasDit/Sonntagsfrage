import yaml
import pandas as pd

import logs
import utils
import configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

def load_data():
    """
        This function loads all data from the sources and combines them into one dataframe for further processing.

        :return: pandas.DataFrame: Returns one dataframe with the combinated data from all input sources.
    """
    logger.info("Started load_data()")

    # open connection to Azure SQL DB
    conn, cursor = utils.connect_to_azure_sql_db()

    # extract Data from Azure SQL DB
    sql_stmt = """select * from sonntagsfrage.results_questionaire_clean"""
    df_survey_results = pd.read_sql(sql_stmt, conn)
    utils.write_df_to_file(df_survey_results, 'load_data_df_survey_results')

    df_all_data_combined = df_survey_results

    return df_all_data_combined



