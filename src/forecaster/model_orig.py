import time
from math import sqrt

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer, SCORERS
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBRegressor

import dataprep as dp
import code_configs
import log

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
logger = log.get_logger(__name__)

TARGET_COLUMN = cfg['model']['target_column']
best_params = code_configs.BEST_MODEL_PARAMS


def date_splitter(p_df, p_train_start, p_start, p_end):
    """
        Diese Funktion splittet ein Dataframe anhand der übergebenen Daten

        :param: p_df (pandas.DataFrame): Hierbei handelt es sich um das zu splittende Dataframe
        :param: p_start (date): Das hier ist das Splitdatum
        :param: p_end (date): Das hier ist das Enddatum bis zu dem die Daten gehen sollen

        :return: Gibt ein Dataframe mit One Hot encoded features zurück
        :rtype: pandas.DataFrame
    """
    logger.debug("Entered date_splitter()")
    before_data = p_df[(p_df.index < p_start) & (p_df.index >= p_train_start)].copy()
    after_data = p_df[(p_df.index >= p_start) & (p_df.index < p_end)].copy()

    return before_data, after_data


def calc_preds(p_train_data_total, p_test_data_total, p_train_mode, p_granularity):
    """
            In dieser Funktion wird die uebergeordnete Logik gekappselt, ob der gesamte Dataframe auf einmal zum
            Trainieren und predicten verwendet wird oder dies pro Welt geschiekt.

            :param p_train_data: pandas.DataFrame; Hierbei handelt es sich um ein Dataframe mit den Trainingsdaten
            :param p_test_data: pandas.DataFrame; Hierbei handelt es sich um ein DataFrame mit den Testdaten
            :param p_train_mode: Gibt an, ob eine gridsearch durchgeführt werden soll. Mögliche Werte lauten: "random", "grid"
                oder "all". Bei "all" Wird keine GridSearch durchgeführt.
            :param p_granularity: Gibt an, ob das Training und das Predicten mit dem gesamten Dataframe statt finden soll,
                oder pro Welt einmal trainiert wird.

            :return: (p_test_data, model): ein Tupel bestehend aus dem ursprünglichen Testdatenset und dem trainierten
            Modell
            :rtype: (pandas.DataFrame, sklearn Estimator)
        """
    logger.debug("Entered train_and_predict()")
    df_with_preds = None

    if p_granularity == 'total':
        df_with_preds, trained_model = train_and_predict(p_train_data_total, p_test_data_total, p_train_mode)

    elif p_granularity == 'ressort':
        filter_ressort_cols = [col for col in p_train_data_total if col.startswith('ressort_')]
        for ressort_col in filter_ressort_cols:
            p_train_data_ressort = p_train_data_total[p_train_data_total[ressort_col] == 1]
            p_test_data_ressort = p_test_data_total[p_test_data_total[ressort_col] == 1]

            p_train_data_ressort = p_train_data_ressort.sort_index()
            p_test_data_ressort = p_test_data_ressort.sort_index()

            # train and predict per ressort
            df_with_preds_ressort, trained_model_ressort = train_and_predict(p_train_data_ressort, p_test_data_ressort,
                                                                             p_train_mode)

            # combine outputs from different ressorts
            frames = [df_with_preds, df_with_preds_ressort]
            df_with_preds = pd.concat(frames)
            trained_model = trained_model_ressort

    return df_with_preds, trained_model


def predict(p_model, p_data):
    """
        Diese Funktion predicted mit dem übergebenen Modell und Daten ohne zu trainieren

        :param: p_model (sklearn Estimator): Hierbei handelt es sich um das bereits trainerte Modell
        :param: p_data (pandas.DataFrame): Hierbei handelt es sich um die Daten (X und y), welche für die Prediciton b
            benötigt werden.

        :return:p_data: pandas.DataFrame; Ein Dataframe mit den Daten und der prediction in der Spalte tsh_predict.
    """
    logger.debug("Entered predict()")
    X = p_data.drop(columns=[TARGET_COLUMN], axis=1).select_dtypes(['number'])

    y_pred = p_model.predict(X)
    p_data[TARGET_COLUMN] = y_pred

    return p_data


def train_and_predict(p_train_data, p_test_data, p_mode):
    """
        Diese Funktion trainiert ein Modell anhand Aufruf der train() Funktion und predicted anhand der übergebenen
        Daten

        :param p_train_data: pandas.DataFrame; Hierbei handelt es sich um ein Dataframe mit den Trainingsdaten
        :param p_test_data: pandas.DataFrame; Hierbei handelt es sich um ein DataFrame mit den Testdaten
        :param p_mode: Gibt an, ob eine gridsearch durchgeführt werden soll. Mögliche Werte lauten: "random", "grid"
            oder "all". Bei "all" Wird keine GridSearch durchgeführt.

        :return: (p_test_data, model): ein Tupel bestehend aus dem ursprünglichen Testdatenset und dem trainierten
        Modell
        :rtype: (pandas.DataFrame, sklearn Estimator)
    """
    logger.debug("Entered train_and_predict()")
    pred_offset_mode = cfg['model']['pred_offset_mode']

    model, X_train, X_test = train(p_train_data, p_test_data, p_mode)

    # X_test = p_test_data.drop(columns=[TARGET_COLUMN], axis=1).select_dtypes(['number'])
    y_test = p_test_data[TARGET_COLUMN]

    y_pred = model.predict(X_test)

    p_test_data[TARGET_COLUMN + '_pred'] = y_pred

    if pred_offset_mode != 'dynamic':
        logger.info("The model is predicting values between " + str(y_test.index.min().date()) + ' and ' + str(
            y_test.index.max().date()))
        logger.info("The RMSE is: " + str(sqrt(mean_squared_error(y_test, y_pred))))
        logger.info("Total number of days predicted is " + str(number_of_days(p_test_data))
                    + " and number of days with a deviation >10% is: " + str(nb_days_over_10_precent(y_test, y_pred)) + '.'
                    )

    return p_test_data, model


def train(p_train_df, p_test_df, p_mode='grid'):
    """
        Diese Funktion trainiert ein Modell anhand eines RandomForestRegressors.

        :param p_test_df: pandas.DataFrame; enthält die Daten (X und y) anhand denen validiert werden soll.
        :param p_train_df: pandas.DataFrame; enthält die Daten (X und y) anhand denen trainiert werden soll.
        :param p_mode: Gibt an, ob eine gridsearch durchgeführt werden soll. Mögliche Werte lauten: "random", "grid"
            oder "all". Bei "all" Wird keine GridSearch durchgeführt.
        :return: model: sklearn Estimator; das trainierte Modell
    """
    now = time.time()
    project = cfg['data_input']['project_name']
    dataset_name = cfg['data_input']['dataset_name']
    feature_importance_table = cfg['model']['features_importance_table']
    features_chosen_by = cfg['feature_engineering']['features_chosen_by']
    pred_offset_mode = cfg['model']['pred_offset_mode']

    logger.debug("Entered train()")
    X_train = p_train_df.drop(columns=[TARGET_COLUMN], axis=1).select_dtypes(['number'])
    y_train = p_train_df[TARGET_COLUMN]
    X_test = p_test_df.drop(columns=[TARGET_COLUMN], axis=1).select_dtypes(['number'])
    y_test = p_test_df[TARGET_COLUMN]

    if features_chosen_by == 'PCA':
        n_components = cfg['feature_engineering']['n_components']
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Save PCA components
        df_components = pd.DataFrame(pca.components_,
                                     columns=list(p_train_df.drop(columns=[TARGET_COLUMN], axis=1).columns)).abs()
        df_components = df_components.sum(axis=0).reset_index().rename(
            columns={'index': 'feature', 0: 'PCA_importance'}).sort_values(by=['PCA_importance'], ascending=False)
        dp.save_df_to_bigquery(df_components, project, dataset_name, feature_importance_table,
                               write_disposition='WRITE_TRUNCATE')

    if p_mode == 'random':
        model = grid_search(X_train, y_train, X_test, y_test, 'random')
    elif p_mode == 'grid':
        model = grid_search(X_train, y_train, X_test, y_test, 'grid')
    else:
        model = XGBRegressor(n_jobs=-1)
        model.set_params(**best_params)
        model.fit(X_train, y_train, eval_metric=custom_valid_function)

    if (features_chosen_by == 'manual') & (pred_offset_mode != 'dynamic'):
        feature_analysis(model, X_train.columns, X_train, y_train, X_test, y_test)
    elif features_chosen_by == 'PCA':
        logger.debug('No. of PCA Components: {}'.format(pca.n_components_))
        logger.debug('No. of features in training: {}'.format(pca.n_features_))

    logger.debug("Training one XGBRegressor took {}s".format(round(time.time() - now, 2)))
    return model, X_train, X_test


def grid_search(p_X_train, p_y_train, p_X_test, p_y_test, p_mode='random'):
    """
        In dieser Funktion wird das Hyperparameter Tuning per GridSearch durchgeführt.

        :param p_X_test: enthält die Daten (X) anhand denen validiert werden soll.
        :param p_y_test: enthält die Daten (y) anhand denen validiert werden soll.
        :param p_X_train: enthält die Daten (X) anhand denen trainiert werden soll.
        :param p_y_train: enthält die Daten (y) anhand denen trainiert werden soll.
        :param p_mode: Gibt das Verfahren an, mit dem die GridSearch durchgeführt werden soll. Mögliche Werte sind
            "random" oder "grid".
        :return: Gibt die optimalen Hyperparameter aus dem Tuning zurück.
    """
    logger.debug("Entered grid_search()")
    global best_params
    tscv = TimeSeriesSplit(n_splits=4)
    param_grid = {
        'nthread': [-1],
        'objective': ['reg:squarederror'],
        "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.7],
        'n_estimators': [100, 200, 300, 500, 1000],
        'num_boost_round': [999],
        'feval': [custom_valid_function]
    }

    SCORERS['nb_days_over_10_precent'] = make_scorer(nb_days_over_10_precent, greater_is_better=False)

    scoring = {
        'RMSE': 'neg_root_mean_squared_error',
        'Days >10%': 'nb_days_over_10_precent'
    }

    fit_params = {
        'eval_set': [[p_X_test, p_y_test]],
        'early_stopping_rounds': 10
    }

    if p_mode == 'random':
        gsc = RandomizedSearchCV(estimator=XGBRegressor(), n_iter=100, param_distributions=param_grid, cv=tscv,
                                 scoring=scoring, refit='Days >10%', n_jobs=-1)
    elif p_mode == 'grid':
        gsc = GridSearchCV(estimator=XGBRegressor(), param_grid=param_grid, cv=tscv, scoring=scoring, refit='Days >10%',
                           n_jobs=-1)

    grid_result = gsc.fit(p_X_train, p_y_train, **fit_params)

    best_params = grid_result.best_params_
    xgbr = XGBRegressor(verbose=True)
    xgbr.set_params(**best_params)
    scores = cross_validate(xgbr, p_X_train, p_y_train, cv=tscv.split(p_X_train), scoring=scoring)
    logger.info("Grid Search Scores: ")
    logger.info(scores)
    logger.info("Grid Search best Params: ")
    logger.info(best_params)
    logger.info("Grid Search best iterations: ")
    logger.info(grid_result.best_estimator_.best_iteration)

    return grid_result.best_estimator_


def feature_analysis(p_model, p_feature_list, p_X_train, p_y_train, p_X_test, p_y_test):
    """
        In ddieser Funktion werden die angefragten Features analysiert und in den Logger werden die Resultate wie
        kumulierte Feature Importance oder Korrelationen weggeschrieben.

        :param p_model: Modell, das zum predicten verwendet wird.
        :param p_feature_list: Liste an Features, die untersucht werden sollen.
        :param p_X_test: enthält die Daten (X) anhand denen validiert werden soll.
        :param p_y_test: enthält die Daten (y) anhand denen validiert werden soll.
        :param p_X_train: enthält die Daten (X) anhand denen trainiert werden soll.
        :param p_y_train: enthält die Daten (y) anhand denen trainiert werden soll.
    """
    logger.debug("Entered feature_analysis()")

    project = cfg['data_input']['project_name']
    dataset_name = cfg['data_input']['dataset_name']
    feature_importance_table = cfg['model']['features_importance_table']

    # Gini Importance
    importances = list(p_model.feature_importances_)

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(p_feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    sorted_importances = [importance[1] for importance in feature_importances]
    sorted_features = [importance[0] for importance in feature_importances]

    df_analysis = pd.DataFrame(feature_importances, columns=['Feature', 'Wert'])

    cumulative_importances = np.cumsum(sorted_importances)
    cumulative_features = [(feature, round(importance, 2)) for feature, importance in
                           zip(sorted_features, cumulative_importances)]
    sorted_perm_features_train = calc_perm_importance(p_X_train, p_model, p_y_train)
    sorted_perm_features_test = calc_perm_importance(p_X_test, p_model, p_y_test)

    df_perm_features_train = pd.DataFrame(sorted_perm_features_train,
                                          columns=['Feature', 'Ind. RMSE', 'Ind. RMSE Std. Abw.'])
    df_perm_features_test = pd.DataFrame(sorted_perm_features_test,
                                         columns=['Feature', 'Acc. RMSE', 'Acc. RMSE Std. Abw.'])

    df_analysis = df_analysis.merge(df_perm_features_train, on='Feature', how='inner')
    df_analysis = df_analysis.merge(df_perm_features_test, on='Feature', how='inner')
    df_analysis.to_pickle(path='./data/df_feat_analysis.pkl', protocol=4)
    dp.save_df_to_bigquery(df_analysis, project, dataset_name, feature_importance_table,
                           write_disposition='WRITE_TRUNCATE')

    logger.debug("Cumulated features:")
    [logger.debug('Variable: {:120} cum. Importance: {}'.format(*pair)) for pair in cumulative_features]

    logger.debug("Individual Importance for Model (in RMSE):")
    [logger.debug('Variable: {:120} Perm. Importance RMSE Mean: {} +/- {}'.format(*triplet)) for triplet in
     sorted_perm_features_train]

    logger.debug("Individual Importance for Accuracy (in RMSE):")
    [logger.debug('Variable: {:120} Perm. Accuracy RMSE Mean: {} +/- {}'.format(*triplet)) for triplet in
     sorted_perm_features_test]

    top_corr = get_top_abs_correlations(p_X_train)
    logger.debug("Top Feature Correlations")
    logger.debug(top_corr)


def calc_perm_importance(p_X, p_model, p_y):
    """
    Fukntion in der die Permutation Importance berechnet wird.

    :param p_X: enthält die Daten (X)
    :param p_model: Modell, das zum predicten verwendet wird.
    :param p_y: enthält die Daten (y)
    :return: list: Gibt eine Liste mit den Werten für die Permutation Importance zurück.
    """
    perm_importance_ = permutation_importance(p_model, p_X, p_y, n_repeats=50, n_jobs=-1,
                                              scoring='neg_root_mean_squared_error')
    perm_means = perm_importance_.importances_mean
    perm_stds = perm_importance_.importances_std
    perm_features = [(feature, round(mean, 2), round(std, 2)) for feature, mean, std in zip(p_X.columns, perm_means,
                                                                                            perm_stds)]
    sorted_perm_features = sorted(perm_features, key=lambda x: x[1], reverse=True)
    return sorted_perm_features


def number_of_days(df):
    """
    Hilfsfunktion, um zu berechnen, wie viele Tage sich in einem Dataframe befinden.

    :param df: Dataframe
    :return: int: Gibt die Anzahl an Tagen zurück, die sich in dem Dataframe befinden.
    """
    delta = df.index.max() - df.index.min()

    return delta.days - 1


def nb_days_over_10_precent(y_test, y_pred):
    """
    Hilfsfunktion um die Anzahl an tagen zu berechnen, bei denen die Abweichung der IST-Werte von der Prediction mehr
    als 10% betragägt.

    :param y_test: enthält die Daten (y) anhand denen validiert werden soll.
    :param y_pred: enthält die Daten (y), die vorhergesagt wurden.
    :return: int: Gibt die Anzahl an Tagen aus, an denen die IST-Werte um mehr als 10% von der prediction abwichen.
    """
    df_tmp = pd.DataFrame(y_test.copy())
    df_tmp['pred'] = y_pred
    df_tmp['rel_abw'] = df_tmp.apply(lambda x: abs(x[TARGET_COLUMN] - x['pred']) / x['pred'], axis=1)
    df_tmp['deviation_is_10'] = df_tmp.apply(lambda x: 1 if x['rel_abw'] > 0.1 else 0, axis=1)
    df_tmp = df_tmp[['deviation_is_10']].reset_index().drop_duplicates(keep='first')

    return df_tmp['deviation_is_10'].sum()


def custom_valid_function(y_test, y_pred):
    """
    Hilffunktion um eine infividuelle Validierungs-Metrik zu berechnen

    :param y_test: enthält die Daten (y) anhand denen validiert werden soll.
    :param y_pred: enthält die Daten (y), die vorhergesagt wurden.
    :return:
    """
    loss = nb_days_over_10_precent(y_test, y_pred)

    return "custom_valid_function", loss, False


def get_redundant_feature_pairs(df):
    """
    Hilfsfunktion um redundante Feature aufzulisten.
    :param df: Dataframe
    :return: set: Gibt ein set aus redundanten Spalten aus.
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=20):
    """
    Funktion um die am stärksten korrekierten Features zu ermitteln.

    :param df: Dataframe
    :param n: Anzahl an auszugebenden Features
    :return: List: Gibt eine Liste mit den n am meisten korrelierten Features aus.
    """
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_feature_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
