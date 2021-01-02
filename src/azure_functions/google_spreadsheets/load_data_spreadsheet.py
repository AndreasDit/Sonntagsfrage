import gspread
import yaml
import os
import sys
import pandas as pd

sys.path.append(os.getcwd())

import utils.logs as logs
import utils.configs_for_code as cfg
import utils.connectivity as conns

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

DATA_SPREADSHEET_NAME = configs['google']['data_spreadsheet_name']
PREDS_SPREADSHEET_NAME = configs['google']['preds_spreadsheet_name']
DATA_WORKSHEET_NAME = configs['google']['data_worksheet_name']
PREDS_WORKSHEET_NAME = configs['google']['preds_worksheet_name']


def empty_worksheet(worksheet):
    """
        Takes one worksheet from a google spreadsheet and deletes all data inside of it.

        :param worksheet: A google worksheet object which represents the actual worksheet that shall be emptied.
    """
    logger.info("Start empty_worksheet()")

    # create empty dummy row: a worksheet can never be truly empty
    row = []
    index = 1
    worksheet.insert_row(row, index)

    # delete the other rows
    nb_rows = worksheet.row_count
    worksheet.delete_rows(start_index=2, end_index=nb_rows + 1)  # index starts at 1 and not at 0


def fill_worksheet_from_df(worksheet, df_input):
    """
        Writes the information inside a dataframe into a google worksheet.

        :param worksheet: A google worksheet object which represents the actual worksheet that shall be filled with data.
        :param df_input: The given dataframe whose data shall be written into a google spreadsheet.
    """
    logger.info("Start fill_worksheet_from_df()")

    # fill worksheet with header line
    all_cols = df_input.columns.values
    header = all_cols.tolist()
    worksheet.insert_row(header, 1)

    # fill worksheet with data lines
    data_rows = []
    for idx in range(2, len(df_input)):  # counting starts at 1 not at 0
        pd_row = df_input.iloc[idx]
        row = pd_row.tolist()
        data_rows.append(row)
    worksheet.insert_rows(data_rows, 2)


def main():
    """
        Main function, performs the data transfer from the Azure SQL DB to a google spreadsheet document..
    """
    logger.info("Start main()")

    # open connections
    conn_azure, cursor_azure = conns.connect_to_azure_sql_db()
    conn_google = conns.connect_to_google_spreadsheets()

    # load worksheets
    sheet_data = conn_google.open(DATA_SPREADSHEET_NAME)
    sheet_preds = conn_google.open(PREDS_SPREADSHEET_NAME)
    data_worksheet = sheet_data.worksheet(DATA_WORKSHEET_NAME)
    preds_worksheet = sheet_preds.worksheet(PREDS_WORKSHEET_NAME)

    # load tables from Azure SQL DB
    sql_stmt = """select * from sonntagsfrage.v_predictions_questionaire_pivot"""
    df_table_preds = pd.read_sql(sql_stmt, conn_azure)
    sql_stmt = """select * from sonntagsfrage.v_results_questionaire_clean_pivot"""
    df_table_data = pd.read_sql(sql_stmt, conn_azure)

    empty_worksheet(data_worksheet)
    empty_worksheet(preds_worksheet)

    fill_worksheet_from_df(data_worksheet, df_table_data)
    fill_worksheet_from_df(preds_worksheet, df_table_preds)


if __name__ == "__main__":
    main()
