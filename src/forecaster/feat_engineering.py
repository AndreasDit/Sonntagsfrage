import yaml
import pandas as pd

import src.forecaster.logs as logs
import src.forecaster.utils as utils
import src.forecaster.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FEATURE_COLS_USED = []
DATE_COL = configs['model']['date_col']
TARGET_COLS = configs['model']['target_cols']

def generate_features(df_input):
    """
            This function generates all features from the base data.

            :return: pandas.DataFrame: Returns a dataframe with all generated features.
    """

    logger.info("Entered generate_features()")

    all_cols = FEATURE_COLS_USED + [DATE_COL] + TARGET_COLS
    df_all_features = df_input[all_cols]

    utils.write_df_to_file(df_all_features, 'generate_features_all_features')

    return df_all_features


