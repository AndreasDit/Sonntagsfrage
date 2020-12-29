from typing import List

import pandas as pd
import yaml
from datetime import datetime, timedelta

import log
import code_configs

logger = log.get_logger(__name__)

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funktion um verschiedene Float-Formate zu harmonisieren.

    :param df: Dataframe, das harmonisiert werden soll
    :return: pandas.Dataframe: Gibt das initial mitgegebene Dataframe zurück mit bereinigten Floats.
    """
    logger.debug("Entered optimize_floats()")

    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funktion um verschiedene int-Formate zu harmonisieren.

    :param df: Dataframe, das harmonisiert werden soll
    :return: pandas.Dataframe: Gibt das initial mitgegebene Dataframe zurück mit bereinigten ints.
    """
    logger.debug("Entered optimize_ints()")

    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    """
    Funktion Spalten im object-Format zu bereinigen.

    :param df: Dataframe, das gereinigt werden soll
    :param datetime_features: Liste an Spaltennamen, die in das Format datetime gebracht werden sollen.
    :return: pandas.Dataframe: Gibt das initial mitgegebene Dataframe zurück mit bereinigten pbject-Spalten.
    """
    logger.debug("Entered optimize_objects()")

    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def optimize(df: pd.DataFrame, datetime_features: List[str] = []):
    """
    Funktion um alle Optimisierungen am Stück auszuführen.

    :param df: Dataframe, das optimiert werden soll.
    :param datetime_features: Liste an Spaltennamen, die in das Format datetime gebracht werden sollen.
    :return: Dataframe mit optimierten Spalten.
    """
    logger.debug("Entered optimize()")

    return optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))


def handle_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Funktion um NaN-Werte zu handeln

    :param df: Dataframe, dessen NaN-Werte bereinigt werden sollen.
    :return: pandas.Dataframe: Dataframe, dessen NaN-Werte bereinigt wurden.
    """
    for col in df.select_dtypes(include=['int32', 'int16', 'int8', 'float32', 'float64']):
        df[col] = df[col].fillna(0)

    return df

def dynamic_train_test_date():
    """ 
        Funktion um Anfangs- und Enddatum der Prognose basierend auf pred_offset_mode zu bestimmen.
        Es ist möglich zwischen zwei Modi zu entscheiden: 
            fixed   :   Die Anfangs- und Enddatum der Prognose sind festgesetzt und im Konfigurationsdatei unter test_start_date 
                        und test_end_date gespeichert.
            dynamic :   Die Anfangs- und Enddatum setzen sich mithilfe der aktuellen Datum (Tag indem man die Prognose durchführt) 
                        und die Variable pred_offset_plus. Die Prognose startet ein Tag nach dem aktuellen Datum und endet in                        
                        pred_offset_minus Tage.
    """
    mode_offset = cfg['model']['pred_offset_mode']
    train_start_date = cfg['model']['train_start_date']

    if mode_offset == 'fixed':
        
        start_date = cfg['model']['test_start_date']
        end_date = cfg['model']['test_end_date']
    
    elif mode_offset == 'dynamic':
        
        offset_plus = cfg['model']['pred_offset_plus']

        start_date = datetime.today()
        end_date = start_date + timedelta(days=offset_plus)
        
    return start_date, end_date
