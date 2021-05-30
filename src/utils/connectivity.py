import pandas as pd
import pyodbc
import yaml
import mysql
import mysql.connector
import gspread
import os
import sys
from datetime import datetime as dt
import logging

sys.path.append(os.getcwd())
import utils.logs as logs
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FILE_PATH_LOGGING = configs['logging']['file_path']
PATH_DATAFRAMES = cfg.PATH_DATAFRAMES
DATE_COL = configs['model']['date_col']
TARGET_COLS = configs['model']['target_cols']
PATH_GOOGLE_SERVICE_ACCOUNT = cfg.PATH_GOOGLE_SERVICE_ACCOUNT


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
        This function creates a connection to the MySQL DB of the final website ai-for-everyone.org.

        :return: connection: Returns the connection to the DB.
        :return: cursor: Returns the cursor which is used to perform database operations on the MySQL DB.
    """
    logger.info('Start connect_to_siteground_sql_db()')

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


def connect_to_google_spreadsheets(auth_type='file', credentials=None):
    """
        This function creates a connection to google spreadsheets using the service account. To access a certain
            document the file must be explicitely shared with this service account via the mail under
            client_email in the 'service_account.json'.

        :return: connection: Returns the connection to google spreadsheets.
    """
    logger.info('Start connect_to_google_spreadsheets()')

    if auth_type == 'file':
        conn = gspread.service_account(filename=PATH_GOOGLE_SERVICE_ACCOUNT)
    elif auth_type == 'env':
        conn = gspread.service_account_from_dict(credentials)

    return conn


def execute_sql_stmt(sql_stmt, cursor, conn):
    cursor.execute(sql_stmt)
    conn.commit()


def write_df_to_sql_db(df_input, conn, cursor, target, header=True, delete_dates=True):
    """
        Writes a dataframe pre multiple single row inserts into an Azure SQL DB. If the target table already has an
            entry for the processed date it gets deleted and overwritten.

        :param df_input: The dataframe with predictions.
        :param conn: Connection to the target DB.
        :param cursor: Cursor to the target DB.
        :param target: Name of the target table.
        :param header: Shall the col names inside the tf be used to insert the data, or shall the data be inserted by position.
    """
    logger.info("Start write_df_to_sql_db() for table " + target)

    pd.options.display.float_format = '{:.5f}'.format

    df_wip = df_input.copy()
    df_string = df_wip.astype(str)
    all_output_col_names = pd.Series(df_string.columns.values)
    logger.info(all_output_col_names)
    # all_output_col_names = pd.Series(['Datum'] + TARGET_COLS)
    header_string = all_output_col_names.str.cat(sep=',')

    # cols = ['Befragte', 'Zeitraum', 'meta_insert_ts']
    # for col in cols:
    #     if col in df_string.columns.values:
    #         df_string[col] = df_string[col].apply(lambda x: "'" + x + "'")

    for col in df_string.columns.values:
        df_string[col] = df_string[col].apply(lambda x: "'" + x + "'")

    if delete_dates == False:
        sqlstmt = """truncate table  """ + target
        logger.info(sqlstmt)
        cursor.execute(sqlstmt)
        conn.commit()


    for idx in range(1, len(df_string)):

        date = df_string.iloc[idx, 0]
        row_as_string = df_string.iloc[idx, 1:].str.cat(sep=',')

        # delete existing row
        sqlstmt = ''
        if delete_dates:
            sqlstmt = """delete from  """ + target + """
                where Datum = """ + date + """ """
            logger.info(sqlstmt)
            cursor.execute(sqlstmt)
            conn.commit()

        # create timestamp
        dt_now = dt.now()
        s_now = dt.strftime(dt_now, '%d.%m.%Y %H:%M:%S')

        # send datarow to azure sql db
        if header: 
            sqlstmt = """insert into """ + target + """( """ + header_string + """, meta_ts )"""
        else: 
            sqlstmt = """insert into """ + target
        sqlstmt += """    values (
            """ + date + """ , """ + row_as_string + """ , '""" + s_now + """'
            )"""
        logger.info(sqlstmt)
        cursor.execute(sqlstmt)
        conn.commit()