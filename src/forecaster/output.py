import yaml
import pandas as pd
import os
import sys
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.getcwd())
import src.forecaster.logs as logs
import src.forecaster.utils as utils
import src.forecaster.configs_for_code as cfg
import src.forecaster.prepare_data as prep
import src.forecaster.feat_engineering as feat

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

TARGET_COLS = configs['model']['target_cols']
DATE_COL = configs['model']['date_col']


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

    df_output = df_working[output_col_names]

    # write to Azure SQL DB
    write_df_to_azure_sql_db(df_output)


def write_df_to_azure_sql_db(df_input):
    """
        Writes a dataframe pre multiple single row inserts into an Azure SQL DB. If the target table already has an
            entry for the processed date it gets deleted and overwritten.

        :param df_input: The dataframe with predictions.
    """
    logger.info("Start export_results()")

    df_wip = df_input
    df_string = df_wip.astype(str)
    all_output_col_names = pd.Series(['Datum'] + TARGET_COLS)
    header_string = all_output_col_names.str.cat(sep=',')

    # open connection
    conn, cursor = utils.connect_to_azure_sql_db()

    for idx in range(1, len(df_string)):

        date = df_string.iloc[idx, 0]
        row_as_string = df_string.iloc[idx, 1:].str.cat(sep=',')

        # delete existing row
        sqlstmt = """delete from  sonntagsfrage.predictions_questionaire
            where Datum = '""" + date + """'"""
        cursor.execute(sqlstmt)
        conn.commit()

        # send datarow to azure sql db
        sqlstmt = """insert into sonntagsfrage.predictions_questionaire( """ + header_string + """ )
            values (
            '""" + date + """' , """ + row_as_string + """
            )"""
        cursor.execute(sqlstmt)
        conn.commit()


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
