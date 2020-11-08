import yaml
import pandas as pd
import os
import sys
from datetime import timedelta

sys.path.append(os.getcwd())
import src.forecaster.logs as logs
import src.forecaster.utils as utils
import src.forecaster.configs_for_code as cfg
import src.forecaster.prepare_data as prep
import src.forecaster.feat_engineering as feat

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

DATE_COL = cfg['model']['date_col']
TARGET_COLS = cfg['model']['target_cols']

def split_df_into_test_train(df_input, train_start_date, test_start_date, test_end_date):
    """
        This function splits a dataframe into test and train sets defined by the given dates.

        :param train_start_date : The starting date of the train set (inclusive).
        :param test_start_date : The starting date of the test set (inclusive).
        :param test_end_date : The ending date of the test set (not inclusive).
        :return:pandas.DataFrame: Returns two dataframes: the train and test dataframe.
    """
    logger.debug("Start split_df_on_given_date()")

    train_data = df_input[(df_input.index < test_start_date) & (df_input.index >= train_start_date)].copy()
    test_data = df_input[(df_input.index >= test_start_date) & (df_input.index < test_end_date)].copy()

    return train_data, test_data


def set_index_if_needed(df_input):
    """
        This function checks if the dataframe has its date column set as a datetime index. If not it sets the
        column (as defined in the configs) as the datetime index.

        :param df_input: The dataframe which does get checked for datetime index.
        :return: pandas.DataFrame: Returns either the input dataframe unmodified or returns the given dataframe
            with its date column set as an index.
    """
    logger.debug("Start set_index_if_needed()")

    ts_idx_exists = 0
    df_tmp = df_input

    if df_tmp.index.dtype != 'datetime64[ns]': ts_idx_exists = 1

    if ts_idx_exists == 0: df_tmp = df_tmp.set_index(DATE_COL)

    df_final = df_tmp
    return df_final


def train_model(df_train, df_test):
    """
        This function performs the training of the model.

        :param df_train: The dataframe with the train data set.
        :param df_test: The dataframe with the test data set.
        :return: model: Returns the trained model which can be used to get predictions.
    """
    logger.debug("Start train_model()")

    X_train = df_train.drop(columns=TARGET_COLS, axis=1).select_dtypes(['number'])
    y_train = df_train[TARGET_COLS]
    X_test = df_test.drop(columns=TARGET_COLS, axis=1).select_dtypes(['number'])
    y_test = df_test[TARGET_COLS]

    return


def get_predictions(df_input):
    """
        This function orchestrates the generation of the predictions. Model definition, trainging and
            predicting take place in this function.

        :param df_input: The dataframe ready for predictions: data is smooth and clean and all features are generated.
        :return: pandas.DataFrame: Returns the dataframe with the predictions.
    """
    logger.debug("Start get_predictions()")

    df_given = df_input
    nb_days_predicted = 1

    df_given = set_index_if_needed(df_given)
    df_given.index = df_given.index.sort_values()

    all_dates_sorted = df_given.index.unique()
    first_date = df_given.index.min()

    # train model with rolling window
    for date in all_dates_sorted[1:]:
        end_date_test_set = date + timedelta(days=1)
        df_train, df_test = split_df_into_test_train(df_given, first_date, date,end_date_test_set)

        model = train_model(df_train, df_test)
