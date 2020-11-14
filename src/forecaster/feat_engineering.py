import yaml
import numpy as np
import pandas as pd

import src.logs as logs
import src.forecaster.utils as utils
import src.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

FEATURE_COLS_USED = [
    # time features
    'day_in_month_sin', 'calendar_week_sin', 'weekday_sin', 'dayofyear_sin', 'month_sin',
    'day_in_month_cos', 'calendar_week_cos', 'weekday_cos', 'dayofyear_cos', 'month_cos',
    'nb_days_since_last_survey'
    ]
DATE_COL = configs['model']['date_col']
TARGET_COLS = configs['model']['target_cols']


def generate_features(df_input):
    """
        This function generates all features from the base data.

        :param df_input: Dataframe with cols that need to be transformed into cyclical representation.
        :return: pandas.DataFrame: Returns a dataframe with all generated features.
    """
    logger.info("Start generate_features()")

    df_no_features = df_input.copy()

    df_added_time_features = create_time_features(df_no_features)

    df_all_features = utils.unset_datecol_as_index_if_needed(df_added_time_features)

    all_cols = FEATURE_COLS_USED + [DATE_COL] + TARGET_COLS
    df_chosen_features = df_all_features[all_cols]

    utils.write_df_to_file(df_chosen_features, 'generate_features_all_features')

    return df_chosen_features


def create_time_features(df_input):
    """
        This function generates all features from the base data.

        :param df_input: Dataframe with cols that need to be transformed into cyclical representation.
        :return: pandas.DataFrame: Returns a dataframe with all generated features.
    """
    logger.info("Start create_time_features()")

    df_wip_time_features = df_input.copy()

    df_wip_time_features = utils.unset_datecol_as_index_if_needed(df_wip_time_features)
    df_wip_time_features['Datum_dt_bckp'] = df_wip_time_features[DATE_COL].astype(str)
    df_wip_time_features_idx = utils.set_datecol_as_index_if_needed(df_wip_time_features)

    # generate simple time features to model seasonal behaviour
    df_wip_time_features_idx['day_in_month'] = df_wip_time_features_idx.index.day
    df_wip_time_features_idx['calendar_week'] = df_wip_time_features_idx.index.isocalendar().week
    df_wip_time_features_idx['weekday'] = df_wip_time_features_idx.index.weekday
    df_wip_time_features_idx['dayofyear'] = df_wip_time_features_idx.index.dayofyear
    df_wip_time_features_idx['month'] = df_wip_time_features_idx.index.month

    # generate cyclical time features to model seasonal behaviour
    df_wip_time_features_idx = make_time_feature_cyclical(df_wip_time_features_idx, 'day_in_month', 30)
    df_wip_time_features_idx = make_time_feature_cyclical(df_wip_time_features_idx, 'dayofyear', 365)
    df_wip_time_features_idx = make_time_feature_cyclical(df_wip_time_features_idx, 'weekday', 7)
    df_wip_time_features_idx = make_time_feature_cyclical(df_wip_time_features_idx, 'calendar_week', 52)
    df_wip_time_features_idx = make_time_feature_cyclical(df_wip_time_features_idx, 'month', 12)

    # generate the number of days since the last survey
    df_wip_nb_days_last = add_number_of_days_since_last_surbey(df_wip_time_features_idx)

    df_all_time_features = df_wip_nb_days_last

    return df_all_time_features


def add_number_of_days_since_last_surbey(df_input):
    """
        Adds the number of days since the last survey as a new feature.

        :param df_input: Dataframe that will get the new feature.
        :return: pandas.Dataframe: Returns the given dataframe with new feature.
    """
    logger.info("Start add_number_of_days_since_last_surbey()")

    df_wip = df_input.copy()
    df_wip['last_survey_date'] = df_wip['Datum_dt_bckp'].shift(1)

    # handle NaN value in first row
    df_min_date = df_wip[df_wip.index == df_wip.index.min()]
    min_date = df_min_date['Datum_dt_bckp'][0]
    values = {'last_survey_date': min_date}
    df_NaN_handled = df_wip.fillna(value=values)

    # calculate time delta as nb of days
    df_NaN_handled['date_last'] = pd.to_datetime(df_NaN_handled['last_survey_date'], format='%Y-%m-%d')
    df_NaN_handled['date_now'] = pd.to_datetime(df_NaN_handled['Datum_dt_bckp'], format='%Y-%m-%d')
    df_NaN_handled['nb_days_since_last_survey'] = df_NaN_handled.apply(lambda x: (x['date_now'] - x['date_last']).days,
                                                                         axis=1)

    df_ouput = df_NaN_handled
    return df_ouput


def make_time_feature_cyclical(df_input, col_for_transform, max_val):
    """
        Transsform time features into cyclidal representation with sin and cos (look polar coordinates for details).

        :param df_input: Dataframe with cols that need to be transformed into cyclical representation.
        :param col_for_transform: Name of the column which needs to be transformed.
        :param max_val: Number of values for one full period: 7 for week, 12 for year etc.
        :return: pandas.Dataframe: Returns the given dataframe with new cyclic column.
    """
    logger.info("Start make_time_feature_cyclical() for col: " + col_for_transform)

    df_col_not_cyclical = df_input.copy()
    df_col_not_cyclical[col_for_transform + '_sin'] = np.sin(2 * np.pi * df_col_not_cyclical[col_for_transform] / max_val)
    df_col_not_cyclical[col_for_transform + '_cos'] = np.cos(2 * np.pi * df_col_not_cyclical[col_for_transform] / max_val)

    df_col_is_cyclical = df_col_not_cyclical
    logger.info("End make_time_feature_cyclical()")
    return df_col_is_cyclical
