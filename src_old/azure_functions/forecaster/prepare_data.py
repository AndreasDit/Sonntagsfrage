import yaml
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta

import utils.logs as logs
import forecaster.utilty as utils
import utils.connectivity as general_utils
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FORECAST_DATE = configs['general']['forecast_date']
FORECAST_MODE = configs['general']['forecast_mode']

def get_forecast_day():
    """
        This function calculates the day that shall be forecasted.
        hard: The date is set to a certain value.
        dynamic: The date is given as a parameter by the orchestration functions (not yet implemented).
        sysdate: The date ist calculated from the sysdate, as in: the next sunday after tha sysdate is chosen.
    """
    final_forecast_date = None

    if FORECAST_MODE == "hard":
        final_forecast_date = FORECAST_DATE
    elif FORECAST_MODE == "sysdate":
        dt_now = dt.now()
        start = dt_now - timedelta(days=dt_now.weekday())
        end = start + timedelta(days=6)
        final_forecast_date = dt.strftime(end, '%d.%m.%Y')
    
    return final_forecast_date

def load_data():
    """
        This function loads all data from the sources and combines them into one dataframe for further processing.

        :return: pandas.DataFrame: Returns one dataframe with the combinated data from all input sources.
    """
    logger.info("Started load_data()")

    df_survey = load_survey_data()

    df_all_data_combined = df_survey

    utils.write_df_to_file(df_all_data_combined, 'load_data_df_all_data_combined')
    return df_all_data_combined


def load_survey_data():
    """
        This function the survey data from the Azure SQL DB.

        :return: pandas.DataFrame: Returns dataframe with servey data.
    """
    logger.info("Started load_survey_data()")

    # open connection to Azure SQL DB
    conn, cursor = general_utils.connect_to_azure_sql_db()

    # get forecast date
    fcast_date = get_forecast_day()

    # extract Data from Azure SQL DB and one dummy line for the day that will be predicted here
    sql_stmt = """
        select * from sonntagsfrage.results_questionaire_clean
        union all
        select '""" + fcast_date + """', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '0', '0', '0'
        """
    df_survey_results = pd.read_sql(sql_stmt, conn)

    df_survey_results_clean = clean_survey_data(df_survey_results)

    df_survey_results_final = df_survey_results_clean
    utils.write_df_to_file(df_survey_results_final, 'load_survey_data')
    return df_survey_results_final


def clean_survey_data(df_input):
    """
    Cleans the data from the survey. Performs all transformation steps.

    :param df_input: the input dataframe with the servey data from the Azure SQL DB
    :return: pandas.DataFrame: Returns dataframe with cleaned servey data.
    """
    logger.info("Started clean_survey_data()")

    df_input_pre_clean = df_input.copy()

    # clean Datum
    df_input_pre_clean['Datum_dt'] = df_input_pre_clean['Datum'].astype(str)
    df_input_pre_clean['Datum_dt'] = df_input_pre_clean['Datum_dt'].str.replace('*', '')
    df_input_pre_clean['Datum_dt'] = df_input_pre_clean['Datum_dt'].apply(lambda x: pd.to_datetime(x) if len(x.split('.')) == 3 else None)
    df_input_pre_clean['Datum_dt'] = pd.to_datetime(df_input_pre_clean['Datum_dt'], format='%d.%m.%Y')
    df_input_clean_Datum = df_input_pre_clean.dropna(subset=['Datum_dt'])

    df_input_clean = df_input_clean_Datum
    return df_input_clean