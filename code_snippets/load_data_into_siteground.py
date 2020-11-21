import pandas as pd
import yaml
import os
import sys
sys.path.append(os.getcwd())

import src.utils.configs_for_code as cfg
import src.utils.connectivity as general_utils
import src.utils.logs as logs

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

WORKSHEET_NAME = configs['google']['worksheet_name']
PATH_GOOGLE_SERVICE_ACCOUNT = cfg.PATH_GOOGLE_SERVICE_ACCOUNT

def copy_table_from_azure_to_website(table_name):
    """
        Copies a table from the Azure SQL DB to the .

        :param table_name: Name of the target table.
    """
    logger.info("Start copy_table_from_azure_to_website()")

    # open connection to Azure SQL DB
    conn_azure, cursor_azure = general_utils.connect_to_azure_sql_db()
    conn_siteground, cursor_siteground = general_utils.connect_to_siteground_sql_db()

    # load table from Azure SQL DB
    sql_stmt = """select * from sonntagsfrage.""" + table_name
    df_table = pd.read_sql(sql_stmt, conn_azure)

    # load data into MySQL DB in website
    general_utils.write_df_to_sql_db(df_table, conn_siteground, cursor_siteground, table_name)


def main():
    """
        Main function, performs the data transfer from the Azure SQL DB to the MySQL DB from website.
    """
    logger.info("Start main()")

    targets = ['results_questionaire_clean', 'predictions_questionaire']

    for target in targets:
        copy_table_from_azure_to_website(target)

    logger.info("Finished copying the following tables: " + targets)


if __name__ == "__main__":
    main()


