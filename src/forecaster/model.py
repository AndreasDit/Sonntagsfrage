import utils.configs_for_code as cfg
import forecaster.evaluation as eval
import forecaster.utilty as utils
import utils.logs as logs
import yaml
import os
import sys
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
from xgboost.sklearn import XGBRegressor
from azureml.core import Run

sys.path.append(os.getcwd())

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)
run = Run.get_context()

TARGET_COLS = configs['model']['target_cols']
LIST_WITH_ALGOS = configs['model']['list_with_algos']


def split_df_into_test_train(df_input, train_start_date, test_start_date, test_end_date):
    """
        This function splits a dataframe into test and train sets defined by the given dates.

        :param df_input : The dataframe which gets split in train and test.
        :param train_start_date : The starting date of the train set (inclusive).
        :param test_start_date : The starting date of the test set (inclusive).
        :param test_end_date : The ending date of the test set (not inclusive).
        :return:pandas.DataFrame: Returns two dataframes: the train and test dataframe.
    """
    # logger.info("Start split_df_on_given_date()")

    train_data = df_input[(df_input.index < test_start_date) & (
        df_input.index >= train_start_date)].copy()
    test_data = df_input[(df_input.index >= test_start_date) & (
        df_input.index < test_end_date)].copy()

    # .select_dtypes(['number'])
    X_train = train_data.drop(columns=TARGET_COLS, axis=1)
    y_train = train_data[TARGET_COLS]
    # .select_dtypes(['number'])
    X_test = test_data.drop(columns=TARGET_COLS, axis=1)
    y_test = test_data[TARGET_COLS]

    return train_data, test_data, X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test, estimator):
    """
        This function performs the training of the model.

        :param df_train: The dataframe with the train data set.
        :param df_test: The dataframe with the test data set.
        :return: model: Returns the trained model which can be used to get predictions.
    """
    logger.info("Start train_model()")

    model = None
    if estimator == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
    if estimator == 'SGDRegressor':
        model = MultiOutputRegressor(SGDRegressor())
        model.fit(X_train, y_train)
    if estimator == 'GradientBoostingRegressor':
        model = MultiOutputRegressor(GradientBoostingRegressor())
        model.fit(X_train, y_train)
    if estimator == 'XGBRegressor':
        best_params = {'colsample_bytree': 0.5, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 5,
                       'min_child_weight': 5, 'n_estimators': 50, 'nthread': -1,
                       #  'num_boost_round': 45,
                       'objective': 'reg:squarederror'}
        model_xgb = XGBRegressor(n_jobs=-1)
        model_xgb.set_params(**best_params)

        model = MultiOutputRegressor(model_xgb)
        model.fit(X_train, y_train)

    return model


def generate_predictions(df_input, estimator):
    """
        This function orchestrates the generation of the predictions. Model definition, trainging and
            predicting take place in this function.


        :param df_input: The dataframe ready for predictions: data is smooth and clean and all features are generated.
        :return: pandas.DataFrame: Returns the dataframe with the predictions.
    """
    logger.info("Start generate_predictions()")

    df_given = df_input.copy()
    nb_days_predicted = 1
    model = None

    # --- handle index
    df_given_idx = utils.set_datecol_as_index_if_needed(df_given)
    all_dates_sorted = df_given_idx.sort_index().index.unique()
    df_for_output_idx = df_given_idx.copy()

    first_date = df_given_idx.index.min()

    # --- train model with rolling window
    logger.info(all_dates_sorted)
    logger.info(range(1, len(all_dates_sorted[1:])))
    for idx in range(1, len(all_dates_sorted[1:])):
        # logger.info("Generate preds for date " + str(date))
        date = all_dates_sorted[idx]
        logger.info(date)
        logger.info(idx)
        logger.info(all_dates_sorted)
        end_date_test_set = date + timedelta(days=nb_days_predicted)
        df_train, df_test, X_train, y_train, X_test, y_test = split_df_into_test_train(df_given_idx, first_date, date,
                                                                                       end_date_test_set)

        model = train_model(X_train, y_train, X_test, y_test, estimator)
        preds_test = model.predict(X_test)
        preds_test = preds_test.round(0)
        # preds_test = [round(num, 0) for num in preds_test]

        # Write preds into output DF
        array_pos = 0
        for party in TARGET_COLS:
            df_for_output_idx.loc[df_for_output_idx.index ==
                                  date, party + '_pred'] = preds_test[0][array_pos]
            array_pos += 1
        df_for_output_idx = df_for_output_idx.fillna(0)

        # Log validation metrics to AML
        cond_last_4_weeks = (df_for_output_idx.index < date) & (
            df_for_output_idx.index >= date-timedelta(weeks=12))
        df_for_logging = df_for_output_idx[cond_last_4_weeks].copy()
        logger.info(df_for_logging)
        eval.get_metrics_for_all_parties(
            df_for_logging, date, estimator)

    df_for_output_idx = df_for_output_idx.fillna(0)

    df_for_output_idx['estimator'] = estimator
    df_given_with_preds = df_for_output_idx

    # --- get corresponding metrics
    predicted_date = max(df_given_with_preds.index)
    df_metrics = eval.get_metrics_for_all_parties(
        df_given_with_preds, predicted_date, estimator)

    return df_given_with_preds, df_metrics


def combined_restults_from_all_algorithms(df_with_features):
    """[summary]

    Args:
        df_with_features ([type]): [description]

    Returns:
        [type]: [description]
    """

    init_algo = LIST_WITH_ALGOS[0]
    df_with_preds, df_metrics = generate_predictions(
        df_with_features, init_algo)

    for idx in range(1, len(LIST_WITH_ALGOS)):
        current_algo = LIST_WITH_ALGOS[idx]
        df_with_preds_tmp, df_metrics_tmp = generate_predictions(
            df_with_features, current_algo)

        frames = [df_with_preds, df_with_preds_tmp]
        df_with_preds = pd.concat(frames)

        frames = [df_metrics, df_metrics_tmp]
        df_metrics = pd.concat(frames)

    utils.write_df_to_file(df_with_preds, 'generate_predictions_finish_preds')
    utils.write_df_to_file(df_metrics, 'generate_predictions_finish_metrics')
    return df_with_preds, df_metrics
