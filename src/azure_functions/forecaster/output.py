import yaml
import os
import sys

from utils.connectivity import write_df_to_sql_db
from utils.connectivity import connect_to_azure_sql_db

sys.path.append(os.getcwd())
import utils.logs as logs
import forecaster.utilty as utils
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

TARGET_COLS = configs['model']['target_cols']
DATE_COL = configs['model']['date_col']
WRITE_TO_AZURE = configs['dev']['write_to_azure']


def export_results(df_input):
    """
        This function writes the generated predictions to variousu sources so that the preds can be consumed py other
            applications.

        :param df_input: The dataframe with predictions.
    """
    logger.info("Start export_results()")

    df_working = df_input.copy()
    df_working = utils.unset_datecol_as_index_if_needed(df_working)
    output_col_names = [DATE_COL] + get_pred_col_names()
    target_table_name = 'sonntagsfrage.predictions_questionaire'

    df_output = df_working[output_col_names]

    # open connection
    conn, cursor = connect_to_azure_sql_db()

    # write to Azure SQL DB
    if WRITE_TO_AZURE: write_df_to_sql_db(df_output, conn, cursor, target_table_name, False)


def export_metrics(df_input):
    """
        This function writes the metrics of the generated predictions to various sources so that the metrics can be
            consumed py other applications.

        :param df_input: The dataframe with predictions.
    """
    logger.info("Start export_results()")

    df_output = df_input.copy()
    target_table_name = 'sonntagsfrage.metric_results'

    # open connection
    conn, cursor = connect_to_azure_sql_db()

    # write to Azure SQL DB
    if WRITE_TO_AZURE: write_df_to_sql_db(df_output, conn, cursor, target_table_name, True)


def get_pred_col_names():
    """
        This function generates the name of the the predicted columns from the given target columns, e.g. SPD_pred from SPD.

        :return: series: Returns a series with the column names of the predicted values.
    """
    logger.info("Start get_pred_col_names()")

    pred_cols = []

    for idx in range(0, len(TARGET_COLS)):
        pred_cols.append(TARGET_COLS[idx] + '_pred')

    return pred_cols
