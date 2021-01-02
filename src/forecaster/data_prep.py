import time
import datetime as dt

import gcsfs
import pandas as pd
import yaml
from google.cloud import bigquery

import code_configs
import helper
import log

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

dataset_name = cfg['data_input']['dataset_name']
project = cfg['data_input']['project_name']
aktionen_manual_transform_cols = {'Status': 'str', 'AKTION': 'str'
        , 'GUTSCHEIN- \r\nCODE': 'str', 'Gutschein-Wert': 'str'
        , 'Werbekanal': 'str', 'stationär einlösbar': 'str'
        , 'Category / Ausschlüsse': 'str', 'Zahlungsmittel': 'str'
        , 'MEW': 'str', 'Gutscheinbedingungen': 'str'
        , 'Einlös-ungen': 'str', 'Bestell-umsatz': 'str'
        , 'Ø WK': 'str', 'stationäre Einlösung': 'str'
        , 'Bewertung Info': 'str', 'check': 'str'
        , 'exkl. Marke': 'str', 'inkl. Marken': 'str'
        , 'Land': 'str', 'Aktionsgröße': 'str'
        , 'Wert': 'str', 'Newsletter Details': 'str'
        , 'Beschränkungen': 'str'
                                  }

logger = log.get_logger(__name__)

def data_prep(p_df):
    """
        Diese Funktion ist der zweite Schritt einer Pipeline. Hier werden die Daten für das Training des Modells bereinigt.
        Dies umfasst das entfernen nichtgebrauchter Spalten, Imputieren von Null Werten und Ändern von Datentypen.

        :param: p_df (pandas.DataFrame): Hierbei handelt es sich um das zu reinigende Dataframe

        :return: pandas.DataFrame: Gibt ein Dataframe aufbereiteter Daten zurück

    """
    logger.debug("Entered data_prep()")
    bigquery_date_col = cfg['data_input']['bigquery_date_col']
    drop_col_list = []

    # TODO replace nan with correct values

    # TODO cast columns to correct type

    # TODO additional lab specific logic

    # set date col as index
    p_df['Datum'] = p_df[bigquery_date_col]
    p_df = p_df.set_index(bigquery_date_col, 'ressort')

    # drop unneeded columns
    p_df = p_df.drop(drop_col_list, axis=1)

    # replace NaN for now with Zeroes
    p_df = helper.handle_na(p_df)

    return p_df


def data_input():
    """
        Diese Funktion ist der erste Schritt der Pipeline und liest die benötigten Daten in Form eines SQL ein.

        Ändere diese Funktion, um zusätzliche Datenquellen anzubinden

        :return: pandas.DataFrame: Gibt ein Dataframe auf Basis des SQL mit deduplizierten Spalten zurück
    """
    logger.debug("Entered data_input()")

    bigquey_table_name = cfg['data_input']['bigquey_table_name']

    mode_bigquery = cfg['data_input']['mode_bigquery']
    sql_file_name = cfg['database']['sql_file_name']
    bigquery_date_col = cfg['data_input']['bigquery_date_col']
    bigquery_pickle_file_name = cfg['data_input']['bigquery_pickle_file_name']
    local_name = bigquery_pickle_file_name + '_' + bigquey_table_name

    einflussfakt_file_name = cfg['data_input']['einflussfakt_file_name']
    einflussfakt_file_date_col = cfg['data_input']['einflussfakt_date_col']
    einflussfakt_file_date_col_format = cfg['data_input']['einflussfakt_date_col_format']

    markting_budget_file_name = cfg['data_input']['markting_budget_file_name']

    aktionen_file_names = cfg['data_input']['aktionen_file_names']

    # Load merketing budget
    df_markting_budget = load_markting_budget_data(markting_budget_file_name)

    # Load einflussfaktoren
    df_einflussfakt_file = load_csv_einflussfaktoren(einflussfakt_file_date_col, einflussfakt_file_date_col_format,
                                                     einflussfakt_file_name)

    # Load aktionen
    df_aktionen_file = load_aktionen_data(aktionen_file_names)

    # Load data from breuningers BigQuery DWH
    logger.debug('start bigquery')
    transform_cols = {'full_date': 'datetime64', 'bestellAbschlussZeitpunkt': 'datetime64'}

    df_bigquery = None
    if mode_bigquery == 'bigquery':
        df_bigquery = read_data_from_bigquery(sql_file_name, transform_cols)
    elif mode_bigquery == 'local':
        df_bigquery = load_df_from_file(local_name, type='pickle')
    elif mode_bigquery == 'gcs':
        bucket_name = 'prognose2020'
        project = "breuninger-playground-adm"
        dataset_id = "Prognosemodell_2020"
        table_id = bigquey_table_name

        export_data_from_bigquery_to_gcs(bucket_name, project, dataset_id, table_id)
        df_bigquery = read_data_from_gcs(project=project, bucket_name=bucket_name, table_id=table_id,
                                         file_name=table_id,
                                         transform_cols=transform_cols, local_name=local_name, parms={})

    # erstelle df mit Datum und Aktionen
    df_date_dim = df_bigquery[[bigquery_date_col]].drop_duplicates().copy()
    df_aktionen_agg = agg_aktion_info_on_date(df_date_dim, df_aktionen_file)
    save_df_to_file(df_aktionen_agg, 'df_aktionen_agg', type='pickle')

    # merge loaded dfs to one
    df = pd.merge(df_bigquery, df_einflussfakt_file, left_on=bigquery_date_col, right_on=einflussfakt_file_date_col,
                  how='left')
    df = pd.merge(df, df_aktionen_agg, left_on=bigquery_date_col, right_on=bigquery_date_col,
                  how='left')
    df = pd.merge(df, df_markting_budget, left_on=['year', 'month'], right_on=['Jahr', 'month_num'], how='left')

    # save to file
    save_df_to_file(df, 'df_dataprep', type='pickle')

    return df


def load_csv_einflussfaktoren(einflussfakt_file_date_col, einflussfakt_file_date_col_format, einflussfakt_file_name):
    """
        Diese Funktion laedt die csv-Datei mit den Einflussfaktoren in ein Dataframe und konvertiert direkt
        die Spalten in ein passendes Format. Zudem werden nas auch behandelt.

        :param einflussfakt_file_date_col: Datumsspalte aus der csv-Datei mit den Einflussfaktoren. Wird spaeter
            zum merge der Dataframes genutzt.
        :param einflussfakt_file_date_col_format: Format in dem die Datumsspalte in der csv-Datei vorliegt, damit
            der Datumsstring in ein date umgewandelt werden kann.
        :param einflussfakt_file_name: Dateiname der csv-Datei mit den Einflussfaktoren.

        :return: pandas.DataFrame: Gibt ein Dataframe mit den Informationen aus der csv-Datei zurueck. Hat bereits
            umgewnadelte Spalten.
    """

    # set parameters
    data_file_drop_columns = ['Notizen']
    parms = {'sep': ';', 'header': 0, 'encoding': 'iso-8859-1', 'decimal': ','}
    bucket_name = cfg['data_input']['bucket_name']

    transform_cols = {'NL Versand': 'int32', 'NL Sale Auflage': 'int32', 'NL nicht Sale mit Gutschein/Coupons': 'int32',
                      'NL nicht Sale Auflage': 'int32'
        , 'Feiertagsdichte': 'float32'
        , 'Feiertag BW': 'int32'
        , 'Reduzierungswelle 1 - alles': 'int32', 'Reduzierungswelle 2 - EX': 'int32'
        , 'Reduzierungswelle 2 - alles': 'int32', 'Reduzierungswelle 3 - Ex': 'int32'
        , 'Aktionstart': 'int32'
        , 'Feriendichte': 'int32', 'Beauty/Schmuck Coupons': 'int32'
        , 'ist F/S': 'int32', 'Private Luxury Sale': 'int32', 'BreuningerBonus': 'int32', 'Coupon': 'int32'
        , 'SportAktion': 'int32', 'FerienBW': 'int32'
                      }
    replace_na_cols = {'NL Versand': 0, 'NL Sale Auflage': 0, 'NL nicht Sale mit Gutschein/Coupons': 0,
                       'NL nicht Sale Auflage': 0
        , 'Feiertagsdichte': 0
        , 'Feiertag BW': 0
        , 'Reduzierungswelle 1 - alles': 0, 'Reduzierungswelle 2 - EX': 0
        , 'Reduzierungswelle 2 - alles': 0, 'Reduzierungswelle 3 - Ex': 0
        , 'Aktionstart': 0
        , 'Feriendichte': 0, 'Beauty/Schmuck Coupons': 0
        , 'ist F/S': 0, 'Private Luxury Sale': 0, 'BreuningerBonus': 0, 'Coupon': 0
        , 'SportAktion': 0, 'FerienBW': 0
                       }

    # load the file
    df_einflussfakt_file = read_data_from_gcs(project=project, bucket_name=bucket_name, table_id=einflussfakt_file_name,
                                              file_name=einflussfakt_file_name, transform_cols=transform_cols,
                                              local_name=einflussfakt_file_name,
                                              parms=parms, drop_columns=data_file_drop_columns)

    # format datetime column
    df_einflussfakt_file[einflussfakt_file_date_col] = pd.to_datetime(df_einflussfakt_file[einflussfakt_file_date_col],
                                                                      format=einflussfakt_file_date_col_format)

    # handle missing values
    df_einflussfakt_file = df_einflussfakt_file.fillna(replace_na_cols)

    return df_einflussfakt_file


def load_markting_budget_data(fname):
    """
        Lädt die csv-Datei mit den Marketingbudgets.

        :param fname: Name der csv-Datei mit dem Marketingbudget.
        :return: pandas.DataFrame: Gibt ein Dataframe mit den Marketingdaten zurück.
    """
    logger.debug("Entered load_markting_budget_data()")

    # set parameters
    project = cfg['data_input']['project_name']
    bucket_name = cfg['data_input']['bucket_name']
    transform_cols = []
    parms = {'sep': ';', 'header': 0, 'encoding': 'iso-8859-1', 'decimal': ',', 'skiprows': 0}

    # load file
    df = read_data_from_gcs(project=project, bucket_name=bucket_name, table_id=fname, file_name=fname,
                            transform_cols=transform_cols,
                            local_name=fname, parms=parms)

    # transform written german month to numeric value
    df['month_num'] = df.apply(lambda x: month_short_to_num(x['Monat']), axis=1)

    # save df to file
    save_df_to_file(df, 'df_load_markting_budget_data', type='pickle')

    return df


def load_aktionen_data(aktionen_file_names):
    """
        Lädt mehrere csv-Dateien über die Aktionen und KonKateniert die resultierenden Dataframes zu einem Datensatz.

        :aktionen_file_names: Liste an Dateiennamen, die eingelesen werden sollen.
        :return: pandas.DataFrame: Gibt ein Dataframe mit den Aktionendaten zurück.
    """
    logger.debug("Entered load_aktionen_data()")

    # set parameters
    project = cfg['data_input']['project_name']
    bucket_name = cfg['data_input']['bucket_name']
    inputs_directory = cfg['data_input']['inputs_directory']
    transform_cols = []
    parms = {'sep': ';', 'header': 0, 'encoding': 'iso-8859-1', 'decimal': ',', 'skiprows': 2}

    # combine seperate files
    df_aktionen_file = None
    for file_name in aktionen_file_names:
        df_tmp = read_data_from_gcs(project=project, bucket_name=bucket_name, table_id=file_name, file_name=file_name,
                                    transform_cols=transform_cols,
                                    local_name=file_name, parms=parms)
        df_for_concat = [df_aktionen_file, df_tmp]
        df_aktionen_file = pd.concat(df_for_concat).drop_duplicates().reset_index(drop=True)

    # delete empty rows with just the month-year information (made for humans to easy navigate the excel file)
    df_aktionen_file = df_aktionen_file.dropna(subset=['Status'])

    # delete rows with unvalid starting date and end date information
    df_aktionen_file = df_aktionen_file.dropna(subset=['Laufzeit der Aktion'])
    df_aktionen_file = df_aktionen_file[df_aktionen_file['Laufzeit der Aktion'] != 'tbd'].copy()

    # generate proper start and end date columns
    df_aktionen_file = extract_dates_from_aktionsfile(df_aktionen_file)

    # save partial result to file
    save_df_to_file(df_aktionen_file, 'df_aktionen_file', type='pickle')
    # df_for_debug = transform_object_type_columns(df_aktionen_file, manual_transform_cols)
    # save_df_to_bigquery(df_for_debug, project, dataset_name, 'dev_dataprep_load_aktionen_data', write_disposition='WRITE_TRUNCATE')

    return df_aktionen_file


def month_short_to_num(month):
    """
        Hilfsfunktion. Wandelt einen abgekürzten Monat in eine Zahl um. Bsp.: Macht aus "Feb" eine 2.

        :param month: Der mit 3 Buchstaben abgekürzte Monat.
        :return: int: Gibt die Nummer des Monats zurück.
    """
    month_num = None

    mapping_dict = dict(Jan=1, Feb=2, Mrz=3, Apr=4, Mai=5, Jun=6, Jul=7, Aug=8, Sep=9, Okt=10, Nov=11, Dez=12)

    month_num = mapping_dict[month]

    return month_num


def agg_aktion_info_on_date(date_dim, df_aktionen_file):
    """
        Aggregiert die Informationen aus der Aktionsplanung und erstellt damit nutzbare Features. So wird zum Beispiel
            die Anzahl an Aktionen einer bestimmten Größe berechnet oder der durchschnittliche Rabattwert.

        :param date_dim: Name der Datumsspalte des Dataframes mit den Informattionen zur Aktionsplanung. Wird für die
            Aggregation verwendet.
        :param df_aktionen_file: Dataframe mit den Informationen zur Aktionsplanung.
        :return: pandas.DataFrame: Gibt das als Input gelieferte Dataframe zurück, bloß um die neuen Spalten mit den Aggregaten erweitert.
    """
    project = cfg['data_input']['project_name']
    save_df_to_file(df_aktionen_file, 'df_aktionen_file', type='pickle')
    date_col = date_dim.columns[0]
    date_dim['key'] = 1
    df_aktionen_file['key'] = 1
    date_dim_with_aktionen = pd.merge(date_dim, df_aktionen_file, on='key', how='outer')
    date_dim_with_aktionen = date_dim_with_aktionen[
        (date_dim_with_aktionen[date_col] >= date_dim_with_aktionen['start_aktion'])
        & (date_dim_with_aktionen[date_col] <= date_dim_with_aktionen['end_aktion'])]

    # prepare aggregation
    date_dim_with_aktionen = date_dim_with_aktionen.set_index(date_col)
    date_dim_with_aktionen['Aktionsgröße'] = date_dim_with_aktionen['Aktionsgröße'].apply(lambda x: str(x).strip())
    date_dim_with_aktionen['groesse_S_bool'] = date_dim_with_aktionen['Aktionsgröße'].apply(
        lambda x: 1 if x in ['s', 'S'] else 0)
    date_dim_with_aktionen['groesse_M_bool'] = date_dim_with_aktionen['Aktionsgröße'].apply(
        lambda x: 1 if x in ['m', 'M'] else 0)
    date_dim_with_aktionen['groesse_L_bool'] = date_dim_with_aktionen['Aktionsgröße'].apply(
        lambda x: 1 if x in ['l', 'L'] else 0)
    date_dim_with_aktionen['groesse_XL_bool'] = date_dim_with_aktionen['Aktionsgröße'].apply(
        lambda x: 1 if x in ['xl', 'XL'] else 0)
    # combine Gutschein-Wert and Wert to one column + clean the new column and convert to useful values
    date_dim_with_aktionen['gutschein_wert_kombiniert'] = date_dim_with_aktionen.apply(
        lambda x: x['Gutschein-Wert'] if x['Wert'] is None else x['Wert'], axis=1)
    date_dim_with_aktionen['gutschein_wert_kombiniert'] = date_dim_with_aktionen['gutschein_wert_kombiniert'].astype(
        'string')
    date_dim_with_aktionen['gutschein_wert_kombiniert'] = date_dim_with_aktionen['gutschein_wert_kombiniert'].apply(
        lambda x: str(x).replace(',', '.'))
    date_dim_with_aktionen['gutschein_wert_kombiniert'] = date_dim_with_aktionen['gutschein_wert_kombiniert'].fillna(
        '0.0')
    date_dim_with_aktionen['gutschein_geldwert'] = date_dim_with_aktionen['gutschein_wert_kombiniert'].apply(
        lambda x: convert_str_to_float_if_possible(str(x).strip()) if convert_str_to_float_if_possible(
            str(x).strip()) >= 1 else 0)
    date_dim_with_aktionen['gutschein_prozentpunkte'] = date_dim_with_aktionen['gutschein_wert_kombiniert'].apply(
        lambda x: convert_str_to_float_if_possible(str(x).strip()) * 100 if convert_str_to_float_if_possible(
            str(x).strip()) < 1 else 0)

    # do aggregation
    date_dim_with_aktionen_agg = date_dim_with_aktionen.groupby([pd.Grouper(freq="D")]).agg(
        anzahl_aktionsgroesse_s=('groesse_S_bool', "sum"),
        anzahl_aktionsgroesse_m=('groesse_M_bool', "sum"),
        anzahl_aktionsgroesse_l=('groesse_L_bool', "sum"),
        anzahl_aktionsgroesse_xl=('groesse_XL_bool', "sum"),
        gutschein_geldwert_mean=('gutschein_geldwert', "mean"),
        gutschein_geldwert_sum=('gutschein_geldwert', "sum"),
        gutschein_geldwert_max=('gutschein_geldwert', "max"),
        gutschein_prozentpunkte_mean=('gutschein_prozentpunkte', "mean"),
        gutschein_prozentpunkte_sum=('gutschein_prozentpunkte', "sum"),
        gutschein_prozentpunkte_max=('gutschein_prozentpunkte', "max"),
        anzahl_aktionen_alle_ressorts=('alle Ressorts', "sum"),
        anzahl_aktionen_ressort_0012=('ressort_0012', "sum"),
        anzahl_aktionen_ressort_0001=('ressort_0001', "sum"),
        anzahl_aktionen_ressort_0011=('ressort_0011', "sum"),
        anzahl_aktionen_ressort_0002=('ressort_0002', "sum"),
        anzahl_aktionen_ressort_0022=('ressort_0022', "sum"),
        anzahl_aktionen_ressort_0003=('ressort_0003', "sum"),
        anzahl_aktionen_ressort_0025=('ressort_0025', "sum"),
        anzahl_aktionen_ressort_0026=('ressort_0026', "sum"),
        anzahl_aktionen_ressort_0006=('ressort_0006', "sum"),
        anzahl_aktionen_ressort_0005=('ressort_0005', "sum"),
        anzahl_aktionen_ressort_0007=('ressort_0007', "sum"),
        anzahl_aktionen_ressort_0009=('ressort_0009', "sum"),
        anzahl_aktionen_ressort_0008=('ressort_0008', "sum"),
    )
    date_dim_with_aktionen_agg = date_dim_with_aktionen_agg.reset_index(level=date_col)
    date_dim_with_aktionen_agg = date_dim_with_aktionen_agg.fillna(0)

    # write restults to table for debugging
    # df_for_debug = transform_object_type_columns(date_dim_with_aktionen_agg, aktionen_manual_transform_cols)
    # save_df_to_bigquery(df_for_debug, project, dataset_name, 'dev_agg_aktion_info_on_date', write_disposition='WRITE_TRUNCATE')

    return date_dim_with_aktionen_agg


def convert_str_to_float_if_possible(p_str):
    """
        Versucht, einen String in einen float zu konvertieren. Stimmt das Format nicht, wird mit dem String nichts gemacht.

        :param p_str: String, der in einen float konvertiert werden soll.
        :return: float: Gibt den zu einem float konvertierten String zurück. Sollte irgendwas an dem String nicht stimmen
            (z.B. falsches Format), so wird statt dessen eine Null zurück gegeben.
    """
    try:
        float(p_str)
        return float(p_str)
    except ValueError:
        return 0


def export_data_from_bigquery_to_gcs(bucket_name, project, dataset_id, table_id):
    """
        Diese Funktion liest eine Tabelle aus der BigQuery aus und kopiert den Inhalt in ein Google Cloud Storage.

        :param bucket_name: Name des Buckets im GCS, in den die Tabelle geschrieben werden soll.
        :param project: Name des Projekts in der Google Cloud.
        :param dataset_id: Name des Schemas, in dem die Tabelle liegt.
        :param table_id: Name der Tabelle, die in das GCS geladen werden soll.
    """
    logger.debug("Export table to Google Cloud Storage")
    client = bigquery.Client(project=project, location="EU")
    inputs_directory = cfg['data_input']['inputs_directory']

    destination_uri = "gs://{}/{}/{}".format(bucket_name, inputs_directory, "{}-*.csv".format(table_id))
    dataset_ref = client.dataset(dataset_id, project=project)
    table_ref = dataset_ref.table(table_id)

    extract_job = client.extract_table(
        table_ref,
        destination_uri,
        # Location must match that of the source table.
        location="EU",
    )  # API request
    extract_job.result()  # Waits for job to complete.

    logger.debug("Exported {}:{}.{} to {}".format(project, dataset_id, table_id, destination_uri))


def read_data_from_gcs(project, bucket_name, table_id, file_name, transform_cols, local_name, parms, drop_columns=None,
                       drop_columns_positions=None):
    """
        Liest Daten aus einem Bucket im Google Cloud Storage (GCS) und wandelt diese in ein Pandas Dataframe um.

        :param bucket_name: Name des Buckets im GCS, in den die Tabelle geschrieben werden soll.
        :param project: Name des Projekts in der Google Cloud.
        :param table_id: Name der Tabelle, die in das GCS geladen werden soll.
        :param file_name: Name der Datei in der Google Cloud Storage, die eingelesen werden soll.
        :param transform_cols: Ein Dict an Spaltennamen und Zielformaten. Alle angegebenen Spalten werden in das
            gewünschte Format umgewandelt.
        :param local_name: Name der Pickle-File, die local zum debuggen abgespeichert wird.
        :param parms: Parameter für die Methode pd.read_csv. Hier sollte ein Dict mitgegeben werden. Es können Parameter
            wie der Seperation Character der csv-Datei angegeben werden oder das Dezimaltrennzeichen bei Zahlen.
        :return: pandas.Dataframe: Gibt einen Dataframe zurück, in den die Daten aus der GCS geladen wurden.
    """
    logger.debug("Read csv from Google Cloud Storage")
    now = time.time()
    gcs = gcsfs.GCSFileSystem(token='google_default')
    inputs_directory = cfg['data_input']['inputs_directory']

    destination_uri = "gs://{}/{}/{}".format(bucket_name, inputs_directory, "{}*.csv".format(table_id))
    files = gcs.glob(destination_uri)

    df = pd.concat([pd.read_csv('gs://' + f, **parms) for f in files], ignore_index=True)
    df = transform_object_type_columns(df, transform_cols)

    if drop_columns_positions is None:
        drop_columns_positions = []

    if drop_columns is None:
        drop_columns = []

    # drop columns via position
    if len(drop_columns_positions) != 0:
        dropped_cols = df.columns[drop_columns_positions]
        df = df.drop(columns=dropped_cols)
    # drop columns via col_name
    if len(drop_columns) != 0:
        df = df.drop(columns=drop_columns)

    df = helper.optimize(df)

    logger.debug("Reading data from Google Cloud Storage took {}s".format(round(time.time() - now, 2)))
    save_df_to_file(df, local_name, type='pickle')
    return df


def read_data_from_bigquery(sql_file_name, transform_cols=None):
    """
        Funktion. doe Datem mit Hilfe eines SQLs aus der BigQuery ausliest und  diese in ein Dataframe umwandelt.

        :param sql_file_name: Name der Datei, in der das SQL abgelegt ist.
        :param transform_cols: Ein Dict an Spaltennamen und Zielformaten. Alle angegebenen Spalten werden in das
                gewünschte Format umgewandelt.
        :return: pandas.Dataframe: Gibt ein Dataframe zurück mit den Daten aus der SQL Abfrage.
    """
    if transform_cols is None:
        transform_cols = []

    logger.debug("Read data from BigQuery")
    logger.debug('start bigquery load from dwh')
    sql = open(sql_file_name, 'rb').read().decode("utf-8")

    sql = sql.replace("<yyyy-mm-dd>", "2020-05-09")

    client = bigquery.Client(project="breuninger-playground-adm", location="EU")
    job = client.query(sql)
    result = job.result()
    df = result.to_dataframe(progress_bar_type='tqdm')

    # transform columns from bigquery
    df = transform_object_type_columns(df, transform_cols)
    # optimize columns
    df = helper.optimize(df)
    # save df for further inspection
    logger.debug('start bigquery save to pickle')
    save_df_to_file(df, 'bigquery', type='pickle')

    return df


def save_df_to_file(df, filename, path='data', type='parquet'):
    """
        Hilfsfunktion, um ein Dataframe local (z.B. zum debuggen) wegzuschreiben.

        :param df: Dataframe, das local gespeichert werden soll.
        :param filename: Dateiname der localen Datei.
        :param path: Pfad, unter dem die Datei gespeichert werden soll.
        :param type: Format, in dem die lokale Datei geschrieben werden soll. Definiert sind "pickle" und  "parquet"
    """
    logger.debug("Save data to local file")
    if type == 'parquet':
        df.to_parquet(f"{path}/{filename}.parquet")
    elif type == 'pickle':
        df.to_pickle(f"{path}/{filename}.pkl", protocol=4)


def save_df_to_gcs(df, filename, bucketname, path=None, type='csv'):
    """
        Hilfsfunktion, um ein Dataframe im Google Cloud Storage zu speichern.

        :param df: Dataframe, der im GCS gespeichert werden soll.
        :param filename: Dateiname der Datei.
        :param path: Pfad, unter dem die Datei gespeichert werden soll.
        :param type: Format, in dem die Datei geschrieben werden soll. Definiert sind 'pickle' und 'csv'
    """
    logger.debug('Save data to GCS')
    if path != None:
        destination_uri = "gs://{}/{}/{}".format(bucketname, path, filename)
    else:
        destination_uri = "gs://{}/{}".format(bucketname, filename)

    if type == 'csv':
        df.to_csv(destination_uri + '.csv')
    elif type == 'pickle':
        df.to_pickle(destination_uri + '.pkl', protocol=4)


def save_df_to_bigquery(df_in, project, dataset_id, table_id, write_disposition='WRITE_APPEND'):
    """
        Hilfsfunktion, um ein Dataframe in BigQuery zu speichern.

        :param df: Dataframe, der im BigQuery gespeichert werden soll.
        :param project: Name des Projektes im Google Cloud.
        :param dataset_id: ID der Datasets im BigQuery.
        :param table_id: ID des Tabelles im Bigquery.
        :param write_disposition: Specifies the action that occurs if the destination table already exists. The following values are supported:
                                  WRITE_TRUNCATE: If the table already exists, BigQuery overwrites the table data and uses the schema from the query result.
                                  WRITE_APPEND: If the table already exists, BigQuery appends the data to the table.
                                  WRITE_EMPTY: If the table already exists and contains data, a 'duplicate' error is returned in the job result.
                                  Default value is WRITE_APPEND.
    """

    logger.debug('Save data to BigQuery')

    # prepare df
    df = df_in.copy()
    replacers = {'\r': '_', 'ä': 'ae',
                 '\n': '_', 'ö': 'oe',
                 ' ': '_', 'ü': 'ue',
                 '-': '_', 'ß': 'ss',
                 '.': '', 'Ø': 'mean_',
                 '/': '_', ':': '_'
                 }
    for key, value in replacers.items():
        df.columns = df.columns.str.replace(key, value)

    # prepare BigQuery client
    client = bigquery.Client(project="breuninger-playground-adm", location="EU")
    dataset_ref = client.dataset(dataset_id, project=project)
    table_ref = dataset_ref.table(table_id)

    # API call                                                                                                                                                                     
    table = client.get_table(table_ref)

    # Configure table
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)

    # API request to insert df into BigQuery
    client.load_table_from_dataframe(df, table, job_config=job_config)


def load_df_from_file(filename, path='data', type='parquet'):
    """
        Lädt die Informationen aus einer Lokalen Datei in ein Dataframe.

        :param filename: Name der lokalen Datei, die geladen werden soll.
        :param path: Pfad, unter dem die zu ladende Datei liegt.
        :param type: Format, in dem die lokale Datei geschrieben wurde. Unterstützt werden zur Zeit "pickle" und  "parquet".
        :return: pandas.Dataframe: Gibt das aus der lokalen Datei geladene Dataframe zurück.
    """
    logger.debug("Read data from local file")

    if type == 'parquet':
        df = pd.read_parquet(f"{path}/{filename}.parquet")
    elif type == 'pickle':
        df = pd.read_pickle(f"{path}/{filename}.pkl")

    df = helper.optimize(df)

    return df


def transform_object_type_columns(df: pd.DataFrame, transform_cols) -> pd.DataFrame:
    """
        Hilfsfunktion, die die Spalten eomes Datafra,es des Typs "object" in ein anderes Format transformiert.
        Bei den Zieldatentypen 'int32', 'int16', 'int8' und 'float32' werden NaN-Werte automatisch mit Nullen
        aufgefüllt.

        :param df: Dataframe, dessen Spalten transformiert werden soll.
        :param transform_cols: Ein Dict an Spaltennamen und Zielformaten. Alle angegebenen Spalten werden in das
                gewünschte Format umgewandelt.
        :return: pandas.Dataframe: Gibt den eingegebenen Dataframe mit transformierten Spaölten zurück.
    """
    logger.debug("Entered transform_columns_from_bigquery()")

    for col in df.select_dtypes(include=['object']):
        if col in transform_cols:
            new_dtype = transform_cols[col]
            if new_dtype in ['int32', 'int16', 'int8', 'float32', 'str']:
                df[col] = df[col].fillna(0)
            df[col] = df[col].astype(new_dtype)
    return df


def extract_dates_from_aktionsfile(df):
    """
        Generiert aus den in der Aktionsplanung angegebenen Zeiträumen eine Datumsspalte "start_aktion" und eine
        Spalte "end_aktion".

        :param df: Dataframe mit den Informationen aus der Aktionsplanung.
        :return: pandas.Dataframe: Gibt das Dataframe mit den Daten aus der Aktionesplanung zurück, das jetzt um
            2 Datumsspalten erweitert wurde.
    """
    df['Laufzeit der Aktion'] = df['Laufzeit der Aktion'].astype('string')

    # split given date interval into start day.month and end day.month
    df['start_aktion_without_year'] = df['Laufzeit der Aktion'].apply(
        lambda x: x.strip()[0:5] if '-' in x else x.strip()[0:5])
    df['end_aktion_with_year'] = df['Laufzeit der Aktion'].apply(
        lambda x: x.split("-")[1].strip()[0:10] if '-' in x else x.strip()[0:10])
    # correct for date inputs like 1.01
    df['start_aktion_without_year'] = df['start_aktion_without_year'].apply(
        lambda x: x[:0] + '0' + x[0:] if x[1] == '.' else x)
    df['end_aktion_with_year'] = df['end_aktion_with_year'].apply(
        lambda x: x[:0] + '0' + x[0:] if x[1] == '.' else x)
    # correct for date inputs like 01.1
    df['start_aktion_without_year'] = df['start_aktion_without_year'].apply(
        lambda x: x[:3] + '0' + x[3:len(x) - 1] if (len(x) == 4) or (x[4] == '.') else x)
    df['end_aktion_with_year'] = df['end_aktion_with_year'].apply(
        lambda x: x[:3] + '0' + x[3:len(x) - 1] if (len(x) == 4) or (x[4] == '.') else x)
    # add year to complete data
    df['start_aktion_with_year'] = df[['start_aktion_without_year', 'Jahr']].apply(
        lambda x: x['start_aktion_without_year'] + '.20' + str(int(x['Jahr'])), axis=1)
    # american format for dates
    df['end_aktion_with_year'] = df['end_aktion_with_year'].apply(
        lambda x: x[6:10] + '-' +  x[3:5] + '-' + x[0:2])
    df['start_aktion_with_year'] = df['start_aktion_with_year'].apply(
        lambda x: x[6:10] + '-' +  x[3:5] + '-' + x[0:2])



    # convert to actual datetime columns
    save_df_to_file(df, 'aktion', type='pickle')
    df['start_aktion'] = df['start_aktion_with_year'].astype('datetime64')
    df['end_aktion'] = df['end_aktion_with_year'].astype('datetime64')

    # save table for debug
    # df_for_debug = transform_object_type_columns(df, aktionen_manual_transform_cols)
    # save_df_to_bigquery(df_for_debug, project, dataset_name, 'dev_dataprep_extract_dates_from_aktionsfile', write_disposition='WRITE_TRUNCATE')

    # drop unneccessary columns
    df = df.drop(columns=['start_aktion_without_year', 'end_aktion_with_year', 'start_aktion_with_year'])

    return df
