import gspread
import yaml
import os
import sys
import pandas as pd

sys.path.append(os.getcwd())

import src.utils.logs as logs
import src.utils.configs_for_code as cfg
import src.utils.connectivity as conns

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

WORKSHEET_NAME = configs['google']['worksheet_name']
SPREADSHEET_NAME = configs['google']['spreadsheet_name']


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

    # load worksheet
    sheet = conn_google.open(SPREADSHEET_NAME)
    worksheet = sheet.worksheet(WORKSHEET_NAME)

    # load table from Azure SQL DB
    sql_stmt = """select * from sonntagsfrage.predictions_questionaire"""
    df_table = pd.read_sql(sql_stmt, conn_azure)

    empty_worksheet(worksheet)

    fill_worksheet_from_df(worksheet, df_table)


if __name__ == "__main__":
    main()
