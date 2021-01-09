import yaml
import os
import sys
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.getcwd())
import utils.logs as logs
import forecaster.utilty as utils
import forecaster.evaluation as eval
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

TARGET_COLS = configs['model']['target_cols']


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

    train_data = df_input[(df_input.index < test_start_date) & (df_input.index >= train_start_date)].copy()
    test_data = df_input[(df_input.index >= test_start_date) & (df_input.index < test_end_date)].copy()

    X_train = train_data.drop(columns=TARGET_COLS, axis=1)  # .select_dtypes(['number'])
    y_train = train_data[TARGET_COLS]
    X_test = test_data.drop(columns=TARGET_COLS, axis=1)  # .select_dtypes(['number'])
    y_test = test_data[TARGET_COLS]

    return train_data, test_data, X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test):
    """
        This function performs the training of the model.

        :param df_train: The dataframe with the train data set.
        :param df_test: The dataframe with the test data set.
        :return: model: Returns the trained model which can be used to get predictions.
    """
    # logger.info("Start train_model()")

    model = DecisionTreeRegressor()
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

    df_given_idx = utils.set_datecol_as_index_if_needed(df_given)
    all_dates_sorted = df_given_idx.index.unique()
    df_for_output_idx = df_given_idx.copy()

    first_date = df_given_idx.index.min()

    # train model with rolling window
    for date in all_dates_sorted[1:]:
        # logger.info("Generate preds for date " + str(date))
        end_date_test_set = date + timedelta(days=nb_days_predicted)
        df_train, df_test, X_train, y_train, X_test, y_test = split_df_into_test_train(df_given_idx, first_date, date,
                                                                                       end_date_test_set)

        model = train_model(X_train, y_train, X_test, y_test)
        preds_test = model.predict(X_test)

        array_pos = 0
        for party in TARGET_COLS:
            df_for_output_idx.loc[df_for_output_idx.index == date, party + '_pred'] = preds_test[0][array_pos]
            array_pos += 1

    # df_for_output = df_for_output.fillna(0)

    df_for_output_idx['estimator'] = estimator
    df_given_with_preds = df_for_output_idx

    # get corresponding metrics
    df_metrics = eval.get_metrics_for_all_parties(df_given_with_preds, estimator)

    utils.write_df_to_file(df_given_with_preds, 'generate_predictions_finish_preds')
    utils.write_df_to_file(df_metrics, 'generate_predictions_finish_metrics')
    return df_given_with_preds, df_metrics
