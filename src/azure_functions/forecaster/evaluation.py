import yaml
import os
import sys
import sklearn.metrics as met
import pandas as pd

from utils.connectivity import write_df_to_sql_db
from utils.connectivity import connect_to_azure_sql_db

sys.path.append(os.getcwd())
import utils.logs as logs
import forecaster.utilty as utils
import utils.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

TARGET_COLS = configs['model']['target_cols']
DATE_COL = configs['model']['date_col']
WRITE_TO_AZURE = configs['dev']['write_to_azure']


def calc_metrics(y_true, y_pred, party, estimator):
    """
        This Function calculates all relevant metrics for model evaluation.

        :param y_true: Array-like list of true values.
        :param y_pred: Array-like list of predicted values.
    """
    logger.info("Start calc_metrics()")

    # calc metrics
    mae = met.mean_absolute_error(y_true, y_pred)
    mse = met.mean_squared_error(y_true, y_pred)
    rmse = met.mean_squared_error(y_true, y_pred, squared=False)
    mape = met.mean_absolute_percentage_error(y_true, y_pred)
    r2 = met.r2_score(y_true, y_pred)

    # combine results into dataframe
    metrics_series = [mae, mse, rmse, mape, r2, party, estimator]

    return metrics_series


def get_metrics_for_all_parties(df_input, estimator):

    df_with_preds = df_input.copy()
    metrics_array = []

    for party in TARGET_COLS:
        y_true_party = df_with_preds.dropna()[party]
        y_pred_party = df_with_preds.dropna()[party+'_pred']
        metrics_series = calc_metrics(y_true_party,y_pred_party,party,estimator)
        metrics_array.append(metrics_series)

    metrics_colnames = ['mae', 'mse', 'rmse', 'mape', 'r2', 'party', 'estimator']
    df_metrics = pd.DataFrame(metrics_array, columns=metrics_colnames)

    return df_metrics
