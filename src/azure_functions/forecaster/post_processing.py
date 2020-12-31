import pandas as pd
import time
import datetime as dt
from math import sqrt

import yaml
from sklearn.metrics import mean_squared_error 

import code_configs
import dataprep as dp
import log
import model

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

logger = log.get_logger(__name__)


def process(p_df, p_export_columns):
    """
    Funnktion um das Dataframe mit den Ergebnissen aus der Prediction für den Export vorzubereiten.

    :param p_df: Dataframe mit Ergebnissen aus der Model-Prediction.
    :param p_export_columns: Liste an Spaltennamen, die ausgegeben werden sollen.
    :return: Dataframe mit allen für die Ausgabe relevanten Informationen.
    """
    logger.debug("Entered process()")
    project = cfg['data_input']['project_name']
    dataset_name = cfg['data_input']['dataset_name']
    rmse_welt_bigquery_table = cfg['post_processing']['rmse_welt_bigquery_table']
    rmse_kw_welt_bigquery_table = cfg['post_processing']['rmse_kw_welt_bigquery_table']
    result_file_path = cfg['main']['result_file_path']
    pred_offset_mode = cfg['model']['pred_offset_mode']
    date = dt.datetime.now() + dt.timedelta(hours=2)
    date = pd.to_datetime(date)

    # Calculate total postprocess-RMSE
    real_pred_df = p_df[['anzahl_positionen', 'anzahl_positionen_pred']].copy()
    real_pred_df = real_pred_df.reset_index()
    real_pred_df = real_pred_df.groupby('Datum').sum()
    if pred_offset_mode != 'dynamic':
        rmse = sqrt(mean_squared_error(real_pred_df['anzahl_positionen'], real_pred_df['anzahl_positionen_pred']))
        logger.info("The total postprocess-RMSE is: " + str(rmse))

        filter_ressort_cols = [col for col in p_df if col.startswith('ressort_')]
        all_cols = filter_ressort_cols + ['anzahl_positionen', 'anzahl_positionen_pred']
        real_pred_ressort = p_df[all_cols].copy()
        real_pred_ressort['kw'] = real_pred_ressort.index.week
        ressort_description = ['DAMENOBERBEKLEIDUNG', 'HERRENKONFEKTION', 'KINDERKONFEKTION', 'SCHUHE', 'PARFUEMERIE',
                               'SPORT', 'WOHNEN', 'WAESCHE / STRUEMPFE', 'DOB EX', 'ACCESSOIRES', 'HAKA EX',
                               'LEDERWAREN / REISEGEPÄCK', 'LUXUS', 'dummy']

        # Calculate postprocess-RMSE of each ressort
        rmse_df = pd.DataFrame()
        for idx, val in enumerate(filter_ressort_cols):
            temp = real_pred_ressort[real_pred_ressort[val] == 1]
            temp = scores(temp).to_frame().transpose()
            temp.insert(0, 'ressort', val)
            temp.insert(1, 'ressort_description', ressort_description[idx])
            rmse_df = rmse_df.append(temp, ignore_index=True)

        rmse_df['postprocess_rmse'] = rmse
        rmse_df.to_csv('./data/rmse_welt.csv', index = False, header=True)
        dp.save_df_to_bigquery(rmse_df, project, dataset_name, rmse_welt_bigquery_table, write_disposition='WRITE_TRUNCATE')

        # Calculate postprocessing-RMSE calendar week vs Ressort
        rmse_df = pd.DataFrame()
        for idx, val in enumerate(filter_ressort_cols):
            temp = real_pred_ressort[real_pred_ressort[val] == 1]
            temp = temp.groupby('kw').apply(scores).reset_index()
            temp.insert(1, 'ressort', val)
            temp.insert(2, 'ressort_description', ressort_description[idx])
            rmse_df = rmse_df.append(temp, ignore_index=True)

        rmse_df.to_csv('./data/rmse_kw_welt.csv', index = False, header=True)
        dp.save_df_to_bigquery(rmse_df, project, dataset_name, rmse_kw_welt_bigquery_table, write_disposition='WRITE_TRUNCATE')

        # Calculate the number of days on which the actual values ​​deviate more from the prediction than 10%
        nb_days = model.nb_days_over_10_precent(real_pred_df['anzahl_positionen'], real_pred_df['anzahl_positionen_pred'])
        logger.info("After postprocessing the number of days with a deviation >10% is: " + str(nb_days))

    # Export prediction
    group_by_key = p_export_columns
    group_by_key.remove('anzahl_positionen_pred')
    p_df = p_df.groupby(group_by_key)[['anzahl_positionen_pred']].sum()

    p_df = p_df.rename(columns={'Datum': 'datum', 'anzahl_positionen_pred': 'anzahl_predicted'})
    p_df['erstellung_ts'] = date

    return p_df


def scores(df):
    """
        Hilfsfunktion um die Auswertung der Prognose zu analysieren.
        anzahl_positionen_avg: Mittelwert der realen Anzahl an Bestellpositionen.
        anzahl_positionen_pred_avg: Mittelwert der prognostizierten Anzahl an Bestellpositionen.
        rmse: Root Mean Squared Error zwischen die realen und die prognostizierten Bestellpositionen.
        rmse_rate: rmse / anzahl_positionen_avg
        abs_error: anzahl_positionen_avg - anzahl_positionen_pred_avg als Absolutwert.
        abs_error_rate: abs_error / anzahl_positionen_avg
        rmse_minus_error = rmse - abs_error

        :param df: Dataframe, der sowohl die realen als auch die prognostizierten Bestellpositionen enthält.
        :return pandas.Series; Array  mit Auswertungen.
    """
    anzahl_positionen_avg = df['anzahl_positionen'].mean()
    anzahl_positionen_pred_avg = df['anzahl_positionen_pred'].mean()
    rmse = sqrt(mean_squared_error(df['anzahl_positionen'], df['anzahl_positionen_pred']))
    rmse_rate = rmse / anzahl_positionen_avg
    abs_error = abs(anzahl_positionen_avg - anzahl_positionen_pred_avg)
    abs_error_rate = abs_error / anzahl_positionen_avg
    rmse_minus_error = rmse - abs_error

    return pd.Series({'rmse_rate': rmse_rate, 'rmse': rmse, 'abs_error': abs_error, 'abs_error_rate': abs_error_rate, 'rmse_minus_error': rmse_minus_error,
                      'anzahl_positionen_avg': anzahl_positionen_avg, 'anzahl_positionen_pred_avg': anzahl_positionen_pred_avg})