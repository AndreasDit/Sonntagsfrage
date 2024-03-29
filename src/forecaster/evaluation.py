import utils.configs_for_code as cfg
import forecaster.utilty as utils
import utils.logs as logs
import yaml
import os
import sys
import sklearn.metrics as met
import pandas as pd
from azureml.core import Run

from utils.connectivity import write_df_to_sql_db
from utils.connectivity import connect_to_azure_sql_db

sys.path.append(os.getcwd())
run = Run.get_context()

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

TARGET_COLS = configs['model']['target_cols']
DATE_COL = configs['model']['date_col']
WRITE_TO_AZURE = configs['dev']['write_to_azure']


def calc_metrics(y_true, y_pred):
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
    metrics_series = [mae, mse, rmse                      # , mape
                      , r2]

    # clean small values, so SQL can parse str to numeric
    metrics_series = [round(num, 4) for num in metrics_series]

    return metrics_series


def get_metrics_for_all_parties(df_input, date, estimator):
    """This function calculates all relevant metrics for this estimator.

    Args:
        df_input ([pandas.dataframe]): Given Dataframe
        estimator ([string]): Algo that was used for generating the predictions.

    Returns:
        [pandas.dataframe]: Gives back a dataframe with all relevant metrics.
    """
    df_with_preds = df_input.copy()
    metrics_array = []

    # --- Calc metrics per party for whole timeframe
    for party in TARGET_COLS:
        y_true_party = df_with_preds.dropna()[party]
        y_pred_party = df_with_preds.dropna()[party+'_pred']
        metrics_series = calc_metrics(y_true_party, y_pred_party)
        metrics_series.append(party)
        metrics_series.append(estimator)
        metrics_array.append(metrics_series)
        # --- Log metrics in experiment
        # logger.info(metrics_array)
        str_date = date.strftime('%Y%m%d')
        # logger.info('str_date')
        # logger.info(str_date)
        run.log_row("Metrics for certain Date, Party and Estimator",
                    date=str_date,
                    mae=metrics_series[0],
                    mse=metrics_series[1],
                    rmse=metrics_series[2],
                    r2=metrics_series[3],
                    party=metrics_series[4],
                    estimator=estimator)
        run.log("date", str_date)
        run.log("mae", metrics_series[0])
        run.log("mse", metrics_series[1])
        run.log("rmse", metrics_series[2])
        run.log("r2", metrics_series[3])
        run.log("party", metrics_series[4])
        run.log("estimator", estimator)

    metrics_colnames = ['mae', 'mse', 'rmse'                        # , 'mape'
                        , 'r2', 'party', 'estimator']
    df_metrics = pd.DataFrame(metrics_array, columns=metrics_colnames)

    # --- Calc metrics over all parties for 2 weeks before the predicted date

    return df_metrics


def calc_and_log_validation_metrics_for_predicted_date(df_input, date, estimator):
    df_wip = df_input.copy()
    test = 1+1
    return 1
