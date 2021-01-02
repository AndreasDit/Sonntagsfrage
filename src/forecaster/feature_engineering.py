import itertools
from datetime import date
from datetime import timedelta

import holidays
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import SelectKBest, f_regression

import code_configs
import log
from helper import dynamic_train_test_date
from dataprep import save_df_to_bigquery

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)
logger = log.get_logger(__name__)
project = cfg['data_input']['project_name']
dataset_name = cfg['data_input']['dataset_name']
feature_importance_table = cfg['main']['feature_bigquery_table']
bigquery_date_col = cfg['data_input']['bigquery_date_col']


def feature_engineering(p_df):
    """
            Diese Funktion führt die verschiedenen Feature Engineering steps nacheinander aus und gibt ein fertiges
            Dataframe zurück

            Returns:
            :return: pandas.DataFrame: Gibt ein Dataframe mit allen Features zurück

    """
    logger.debug("Entered feature_engineering()")

    target_col = cfg['model']['target_column']

    # choose which features to use for the algorithm
    feature_cols = [target_col
        , 'anzahl_bestellungen', 'Datum', 'year', 'year_week', 'year_day', 'month', 'week_day', 'day_is_weekday'
        , 'anzahl_aktionsgroesse_s', 'anzahl_aktionsgroesse_m', 'anzahl_aktionsgroesse_l', 'anzahl_aktionsgroesse_xl'
        , 'gutschein_geldwert_mean', 'gutschein_geldwert_sum', 'gutschein_geldwert_max', 'gutschein_prozentpunkte_mean'
        , 'gutschein_prozentpunkte_sum', 'gutschein_prozentpunkte_max', 'CyberSale', 'weihnachtszeit', 'Marketingbudget'
        , 'cost', 'NL Versand', 'NL Sale Auflage', 'NL nicht Sale mit Gutschein/Coupons', 'NL nicht Sale Auflage'
        , 'plan_umsatz', 'avg_price'
        , 'Feiertagsdichte', 'Feiertag BW', 'Reduzierungswelle 1 alles', 'Reduzierungswelle 2 EX'
        , 'Reduzierungswelle 2 alles', 'Reduzierungswelle 3 Ex', 'Aktionstart', 'Feriendichte'
        , 'Beauty/Schmuck Coupons', 'ist F/S', 'Private Luxury Sale', 'BreuningerBonus'
        , 'Coupon', 'SportAktion', 'FerienBW'
        , 'ressort', 'anzahl_retouren'
        , 'total_abs_reduk', 'total_rel_reduk', 'max_abs_reduk', 'max_rel_reduk', 'min_abs_reduk'
        , 'min_rel_reduk', 'avg_abs_reduk', 'avg_rel_reduk', 'stddev_abs_reduk', 'stddev_rel_reduk'
        , 'anzahl_aktionen_alle_ressorts'        , 'anzahl_aktionen_ressort_0012'        , 'anzahl_aktionen_ressort_0001'        , 'anzahl_aktionen_ressort_0011'
        , 'anzahl_aktionen_ressort_0002'        , 'anzahl_aktionen_ressort_0022'        , 'anzahl_aktionen_ressort_0003'        , 'anzahl_aktionen_ressort_0025'
        , 'anzahl_aktionen_ressort_0026'        , 'anzahl_aktionen_ressort_0006'        , 'anzahl_aktionen_ressort_0005'        , 'anzahl_aktionen_ressort_0007'
        , 'anzahl_aktionen_ressort_0009'        , 'anzahl_aktionen_ressort_0008'
         ]

    # rename cols for write to bigquery
    p_df = p_df.rename(columns={'Reduzierungswelle 1 - alles': 'Reduzierungswelle 1 alles'
        , 'Reduzierungswelle 3 - Ex': 'Reduzierungswelle 3 Ex'
        , 'Reduzierungswelle 2 - EX': 'Reduzierungswelle 2 EX'
        , 'Reduzierungswelle 2 - alles': 'Reduzierungswelle 2 alles'
                                })

    total_cols_for_model = feature_cols
    p_df = p_df[total_cols_for_model]

    # add features
    logger.debug("Entered feature_engineering()")
    p_df = build_time_avg_features(p_df)
    p_df = build_holiday_features(p_df)
    p_df = build_calendar_features(p_df)
    p_df = build_business_features(p_df)
    p_df = build_onehot_features(p_df)
    p_df = build_saison_features(p_df)

    final_features = p_df.columns.tolist()
    final_features.remove(target_col)

    p_df = p_df.fillna({'ressort': 'dummy'})
    p_df = p_df.fillna(0)
    p_df = p_df.groupby(final_features, as_index=False)[target_col].sum()
    p_df = p_df.set_index('Datum')

    correlated_features = ['agg_kw_day_anzahl_bestellungen_max', 'agg_dayofyear_anzahl_bestellungen_mean',
                           'agg_kw_day_anzahl_bestellungen_mean',
                           'agg_dayofyear_anzahl_bestellungen_max',
                           'agg_dayofyear_anzahl_bestellungen_min', 'agg_kw_day_anzahl_positionen_min',
                           'calendar_week_sin', 'calendar_week_cos', 'gutschein_geldwert_sum', 'month_sin', 'month_cos',
                           'year', 'agg_kw_day_anzahl_positionen_max', 'gutschein_prozentpunkte_sum',
                           'agg_dayofyear_anzahl_positionen_max', 'gutschein_prozentpunkte_max',
                           'gutschein_geldwert_max', 'agg_rolling_anzahl_bestellungen', 'agg_kw_day_markting_cost_max',
                           'agg_kw_day_markting_cost_max', 'avg_dayofyear_plan_positionen',
                           'agg_week_day_avg_price_max', 'agg_week_day_anzahl_positionen_min',
                           'agg_dayofyear_avg_price_mean', 'agg_kw_day_avg_price_mean', 'agg_week_day_avg_price_mean',
                           'agg_week_day_markting_cost_mean', 'agg_week_day_anzahl_positionen_mean', 'plan_umsatz',
                           'agg_week_day_anzahl_positionen_max', 'next_ostern', 'last_weihnachten', 'last_ostern',
                           'next_weihnachten', 'Feiertag BW'
                           , 'agg_dayofyear_avg_price_min'
                           , 'agg_kw_day_avg_price_min'
                           ]

    unimportant_features = ['agg_year_month_cost_cumsum', 'gutschein_geldwert_mean', 'anzahl_aktionsgroesse_m',
                            'anzahl_aktionsgroesse_s']

    bad_features = ['agg_dayofyear_anzahl_positionen_mean', 'agg_dayofyear_anzahl_positionen_min',
                    'gutschein_prozentpunkte_mean', 'lastHoliday', 'agg_rolling_anzahl_positionen',
                    'agg_dayofyear_markting_cost_mean', 'day_in_month_sin', 'NL_nicht Sale Auflage', 'Nov',
                    'agg_dayofyear_markting_cost_max', 'agg_kw_day_markting_cost_mean',
                    'agg_kw_day_anzahl_bestellungen_min', 'nextHoliday', 'Sunday', 'weekday_sin']

    features_with_leakage = ['cost']
    calendar_feat = ['day_in_month', 'calendar_week', 'week_day', 'dayofyear', 'year_week', 'year_day', 'month']

    features_chosen_by = cfg['feature_engineering']['features_chosen_by']

    if features_chosen_by == 'manual':
        drop_feat_lists = [correlated_features, calendar_feat, features_with_leakage, bad_features,
                           unimportant_features]
    elif features_chosen_by == 'PCA':
        drop_feat_lists = [features_with_leakage]

    drop_col = list(itertools.chain(*drop_feat_lists))
    df_chosen_feat = p_df[p_df.columns.difference(drop_col)].copy()
    TARGET_COLUMN = cfg['model']['target_column']

    X = df_chosen_feat.drop(columns=[TARGET_COLUMN], axis=1).select_dtypes(['number'])
    y = df_chosen_feat[TARGET_COLUMN]

    skb_k = cfg['feature_engineering']['selectkbest']
    skb = SelectKBest(f_regression, k=skb_k)
    skb.fit(X, y)

    map = zip(skb.get_support(), df_chosen_feat.columns)
    chosen_feat = [TARGET_COLUMN]

    for bool, feature in map:
        if bool:
            chosen_feat.append(feature)

    chosen_feat = [TARGET_COLUMN, 'CyberSale', 'agg_kw_day_anzahl_positionen_mean', 'NL Sale Auflage', 'avg_price',
                   'agg_week_day_anzahl_bestellungen_max', 'anzahl_aktionsgroesse_l', 'dayofyear_sin',
                   'agg_week_day_avg_price_min'
                    , 'Marketingbudget',
                   'agg_week_day_anzahl_bestellungen_mean', 'last_holiday', 'NL Versand',
                   'Tuesday', 'Apr', 'Aug', 'Feb', 'Jul', 'Jun', 'Mar', 'May', 'Monday',
                   'NL nicht Sale mit Gutschein/Coupons', 'Oct', 'Saturday', 'Sep', 'Thursday', 'Wednesday',
                   'agg_dayofyear_markting_cost_min', 'agg_kw_day_markting_cost_min', 'agg_week_day_markting_cost_max',
                   'agg_week_day_markting_cost_min', 'anzahl_aktionsgroesse_xl', 'holiday_fg', 'Jan', 'ostern_fg',
                   'weekend', 'weihnachten_fg', 'Friday', 'day_is_weekday', 'agg_kw_day_avg_price_max', 'weekday_cos',
                   'agg_week_day_anzahl_bestellungen_min', 'day_in_month_cos',
                   'Dec'
                    , 'agg_dayofyear_avg_price_max'
                    , 'next_holiday', 'saison_FS', 'saison_HW',
                   'NL nicht Sale Auflage', 'avg_kw_day_plan_positionen', 'dayofyear_cos'
                    , 'agg_kw_day_total_abs_reduk_mean'
                   # , 'agg_kw_day_total_abs_reduk_max', 'agg_kw_day_total_abs_reduk_min'
                    , 'agg_kw_day_total_rel_reduk_mean'
                    # ,'agg_week_day_total_rel_reduk_mean'
                    ,'agg_week_day_stddev_rel_reduk_min'
                    , 'agg_week_day_stddev_rel_reduk_mean'
                   # , 'agg_week_day_avg_rel_reduk_mean'
                   #  ,'agg_rolling_avg_abs_reduk'
                   #  ,'agg_rolling_avg_rel_reduk'
                   , 'anzahl_aktionen_alle_ressorts'
                   , 'anzahl_aktionen_ressort_0012', 'anzahl_aktionen_ressort_0001', 'anzahl_aktionen_ressort_0011'
                   , 'anzahl_aktionen_ressort_0002', 'anzahl_aktionen_ressort_0022', 'anzahl_aktionen_ressort_0003'
                   , 'anzahl_aktionen_ressort_0025', 'anzahl_aktionen_ressort_0026', 'anzahl_aktionen_ressort_0006'
                   , 'anzahl_aktionen_ressort_0005', 'anzahl_aktionen_ressort_0007'
                   , 'anzahl_aktionen_ressort_0009', 'anzahl_aktionen_ressort_0008'
                   , 'anzahl_aktionen_aktuelles_ressort'
                   ]

    # add one hot encoded ressort categories
    for col in df_chosen_feat.columns:
        if col[0:7] == 'ressort':
            chosen_feat.append(col)
    return df_chosen_feat[chosen_feat]
    # return df_chosen_feat[list(dict.fromkeys(chosen_feat))]


def build_business_features(p_df):
    """
        Funktion, die Business-relevante Features generiertl.

        :param p_df: Dataframe, der um Business-Features angereichert werden soll.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um die Business Features erweitert.
    """
    logger.debug("Entered build_calendar_features()")

    # Berechne Geplante Positionen aus dem geplanten Umsatz
    p_df.loc[:, 'avg_kw_day_plan_positionen'] = p_df['plan_umsatz'] / p_df['agg_kw_day_avg_price_mean']
    p_df.loc[:, 'avg_dayofyear_plan_positionen'] = p_df['plan_umsatz'] / p_df['agg_dayofyear_avg_price_mean']

    # Gib an, wie viele geplante Aktionen im eigenen Ressort statt finden
    all_ressorts = p_df['ressort'].fillna('dummy').drop_duplicates()
    for ressort in all_ressorts:

        if ressort == 'dummy' or ressort is None or ressort == 'nan':
            continue

        feature_ressort_col = 'anzahl_aktionen_ressort_' + str(ressort)
        filter_condition_ressort = p_df.ressort == ressort
        p_df_ressort = p_df[filter_condition_ressort]

        p_df.loc[filter_condition_ressort, 'anzahl_aktionen_aktuelles_ressort'] = p_df_ressort[feature_ressort_col]

    # save output for debugging
    # save_df_to_bigquery(p_df, project, dataset_name, feature_importance_table+'_business', write_disposition='WRITE_TRUNCATE')

    return p_df


def build_onehot_features(p_df):
    """
        Diese Funktion baut die benötigten onehot encoded features

        :param: p_df: Dataframe mit Features, die onehot encoded werden sollen.
        :return: pandas.DataFrame: Gibt ein Dataframe mit One Hot encoded features zurück

    """
    logger.debug("Entered build_onehot_features()")
    categorical_feat_list = cfg['feature_engineering']['one_hot_features']

    # use row index for join
    idx_name = p_df.index.name
    p_df = p_df.reset_index()

    one_hot = pd.get_dummies(p_df[categorical_feat_list])
    p_df = p_df.join(one_hot)
    p_df = p_df.drop(categorical_feat_list, axis=1)

    p_df = p_df.set_index(bigquery_date_col)

    # save output for debugging
    # save_df_to_bigquery(p_df, project, dataset_name, feature_importance_table+'_onehot', write_disposition='WRITE_TRUNCATE')

    return p_df


def build_holiday_features(p_df):
    """
        Funktion, die Feiertags Features generiertl.

        :param p_df: Dataframe, der um Feiertags angereichert werden soll.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um die Feiertags erweitert.
    """
    logger.debug("Entered build_holiday_features()")

    countries_list = cfg['feature_engineering']['countries']

    start_date, end_date = dynamic_train_test_date()
    years = np.arange(2015, end_date.year + 1).tolist()

    holidays_list = [holidays.CountryHoliday(country, years=years).items() for country in countries_list]

    # reset index
    p_df = p_df.reset_index(level=bigquery_date_col)

    # This way is not working anyhow i try --> holiday_fg is always 0 cuz given date is NEVER in holidays_list
    # p_df['holiday_fg'] = [1 if date(row.date().year,row.date().month,row.date().day) in holidays_list else 0 for row in p_df['Datum']]
    # Work around withotu countries:
    holidays_list = holidays.CountryHoliday('DE', years=[2015, 2016, 2017, 2018, 2019, 2020])

    # add holiday flag, days to next holiday and from last holiday
    p_df = add_features_for_given_holidays(bigquery_date_col, holidays_list, p_df, 'holiday')

    # treat x-mas individually
    filter_holidays = ['Erster Weihnachtstag', 'Zweiter Weihnachtstag']
    xmas_holidays_list = filter_dict_by_value(holidays_list, filter_holidays)
    p_df = add_features_for_given_holidays(bigquery_date_col, xmas_holidays_list, p_df, 'weihnachten')

    # treat easter holidays individually
    filter_holidays = ['Karfreitag', 'Ostermontag']
    xmas_holidays_list = filter_dict_by_value(holidays_list, filter_holidays)
    p_df = add_features_for_given_holidays(bigquery_date_col, xmas_holidays_list, p_df, 'ostern')

    # set index
    p_df = p_df.set_index(bigquery_date_col)

    # save output for debugging
    # save_df_to_bigquery(p_df, project, dataset_name, feature_importance_table+'_holiday', write_disposition='WRITE_TRUNCATE')

    return p_df


def filter_dict_by_value(dictObj, filt_values):
    """
    Hilfsfunktion um ein Dict nach einem bestimmten Value zu filtern. KEIN key-value pair, sondern nur nach einem ganz
    bestimmten value.

    :param dictObj: Dict, das gefiltert werden soll.
    :param filt_values: Liste mit Values, nach denen gefiltert werden soll.
    :return: Gibt das gefilterte Dict zurück.
    """
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if value in filt_values:
            newDict[key] = value
    return newDict


def add_features_for_given_holidays(bigquery_date_col, holidays_list, p_df, base_colname):
    """
        Hilfesfunktion um aus den mitgegebenen Feiertagen weitere Features wie z.B. "Tage seit letztem Feiertag" oder
        "Tage bis zum nöchsten Feiertag" zu generieren.

        :param p_df: Dataframe mit Feiertagen, aus denen neue Features berechnet werden sollen.
        :param bigquery_date_col: Neme der Datumsspalte.
        :param holidays_list: Ein Dict mit Feiertagen im Format {Name_des_feiertags:Datum}.
        :param base_colname: Name der Spalte, in der das Flag steht, ob es sich bei einem Tag um eunen Feiertag
            handelt oder nicht.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe mit Feiergagen zurück, aber um die
            neuen Features erweitert.
    """
    # generate column names
    colname_flag = base_colname + '_fg'
    colname_lastHoliday = 'last_' + base_colname
    colname_nextHoliday = 'next_' + base_colname

    # add flag marking a holiday
    p_df[colname_flag] = [1 if date(row.date().year, row.date().month, row.date().day) in holidays_list else 0 for row
                          in p_df[bigquery_date_col]]
    holidays_df = p_df[p_df['holiday_fg'] > 0].drop_duplicates(bigquery_date_col)
    date_df = pd.DataFrame(p_df[bigquery_date_col].drop_duplicates())

    # add days to and days from a holiday
    date_df['holidayTuple'] = [calc_diff_holiday(row, holidays_df) for row in date_df.iterrows()]
    date_df[[colname_lastHoliday, colname_nextHoliday]] = pd.DataFrame(date_df['holidayTuple'].tolist(),
                                                                       index=date_df.index)
    date_df = date_df.drop('holidayTuple', axis=1)

    # merge holiday info onto existing df
    p_df = p_df.merge(date_df, on=bigquery_date_col, how='left')
    p_df = p_df.set_index(bigquery_date_col)
    p_df = p_df.reset_index(level=bigquery_date_col)

    # handle missing values
    p_df = p_df.dropna(subset=[colname_lastHoliday, colname_nextHoliday])

    return p_df


def cyclical_encode(p_df, p_col, p_max_val):
    """
        Hilfsfunktion um eine Datumsspalte in eine zyklische Repräsentation mit sin und cos umzuwandeln.
        Für Nähere Erläuterunge bitte als Beispiel das "Wochenrad" und "Ploar Koordinaten" nachschlagen.

        :param p_df: Dataframe, der um zyklische Zeitkoordinaten erweitert werden soll.
        :param p_col: Name der Spalte, aus der die zyklischen Koordinaten generiert werden sollen.
        :param p_max_val: Anzahl an Werten, nach denen eine zeitliche Periode durchgelaufen ist. Bsp.: handelt es sich um
            Wochen dann 7, bei Jahren dann 12.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um die zyklischen Zeit-Features
            erweitert.
    """
    p_df[p_col + '_sin'] = np.sin(2 * np.pi * p_df[p_col] / p_max_val)
    p_df[p_col + '_cos'] = np.cos(2 * np.pi * p_df[p_col] / p_max_val)

    return p_df


def build_calendar_features(p_df):
    """
        Funktione, die weitere relevante Zeitinformationen aus der in der config-Datei definierten Datumsspalte
            generiert.

        :param p_df: Dataframe, das um weitere Zeit Informationen erweitert werden soll.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um weitere Zeit-Features
            erweitert.
    """
    logger.debug("Entered build_calendar_features()")
    bigquery_date_col = cfg['data_input']['bigquery_date_col']

    # columns from index
    p_df['day_in_month'] = p_df.index.day
    p_df['calendar_week'] = p_df.index.week
    p_df['weekday'] = p_df.index.weekday
    p_df['dayofyear'] = p_df.index.dayofyear
    p_df['month'] = p_df.index.month
    drop_col = ['day_in_month', 'calendar_week', 'weekday', 'dayofyear', 'month']

    p_df = cyclical_encode(p_df, 'day_in_month', 30)
    p_df = cyclical_encode(p_df, 'dayofyear', 365)
    p_df = cyclical_encode(p_df, 'weekday', 7)
    p_df = cyclical_encode(p_df, 'calendar_week', 52)
    p_df = cyclical_encode(p_df, 'month', 12)

    # prepare index
    idx_name = p_df.index.name
    p_df = p_df.reset_index()

    one_hot_weekdays = pd.get_dummies(p_df['weekday'], prefix='weekday')
    p_df = p_df.join(one_hot_weekdays)
    p_df = p_df.rename(columns={'weekday_0': 'Monday', 'weekday_1': 'Tuesday', 'weekday_2': 'Wednesday',
                                'weekday_3': 'Thursday', 'weekday_4': 'Friday', 'weekday_5': 'Saturday',
                                'weekday_6': 'Sunday'})

    one_hot_months = pd.get_dummies(p_df['month'], prefix='month')
    p_df = p_df.join(one_hot_months)
    p_df = p_df.rename(columns={'month_1': 'Jan', 'month_2': 'Feb', 'month_3': 'Mar',
                                'month_4': 'Apr', 'month_5': 'May', 'month_6': 'Jun',
                                'month_7': 'Jul', 'month_8': 'Aug', 'month_9': 'Sep', 'month_10': 'Oct',
                                'month_11': 'Nov', 'month_12': 'Dec'})
    p_df = p_df.set_index(idx_name)

    # columns from date column
    p_df = p_df.reset_index(level=bigquery_date_col)
    p_df['weekend'] = p_df[bigquery_date_col].apply(lambda x: 1 if x.weekday in [5, 6] else 0)
    p_df = p_df.set_index(bigquery_date_col)

    p_df = p_df[p_df.columns.difference(drop_col)].copy()

    # save output for debugging
    # save_df_to_bigquery(p_df, project, dataset_name, feature_importance_table+'_calendar', write_disposition='WRITE_TRUNCATE')

    return p_df


def calc_diff_holiday(p_date, holidays_df):
    """
        Funktion bekommt eine Zeile eines Dataframes, in welcher ein Datum steht sowie ein Dataframe mit allen vorhandenen Feiertagen.
        Berechnet die Differenz zwischen dem Input Datum und den vorhandenen Feiertagen.
        Gibt zurück wie viele Tage seit dem letzten Feiertag vergangen sind und wann der nächste Feiertag ist.

        :param: p_date: Zeile aus einem Dataframe.
        :param: holidays_df: Dataframe mit Feiertagen.
        :return: int, int: Gibt als Integer die Tage seit dem letzten Feiertag und Die Anzahl an Tagen bis zum nöchsten Feiertag zurück.
    """

    # logger.debug("Entered calc_diff_holiday()")
    bigquery_date_col = cfg['data_input']['bigquery_date_col']

    liste = np.asarray([(holiday - p_date[1][0]) / timedelta(days=1) for holiday in holidays_df[bigquery_date_col]])

    lastHoliday = max(liste[liste < 0], default=np.nan)
    nextHoliday = min(liste[liste > 0], default=np.nan)

    return lastHoliday, nextHoliday


def build_time_avg_features(df):
    """
        In dieser Funktion werden alle Features generiert, die darauf basieren, dass Durchschnittswerte über die Zeit
        gebildet werden. Z.B. werden aus der historischen Anzahl an Bestellpositionen pro KW und Wochentag mean, min
        und max generiert.

        :param p_df: Dataframe, das um die Zeit-Durchschnitts Features erweitert werden soll.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um weitere Zeit-Durchschnitts
            Features erweitert.
    """
    # decide on the interesting features
    drop_col = ['anzahl_bestellungen', 'dayofyear', 'year', 'month', 'day_in_month']
    test_start_date, test_end_date = dynamic_train_test_date()

    # calc time dimensions
    df.loc[:, 'dayofyear'] = df.index.dayofyear
    df.loc[:, 'year'] = df.index.year
    df.loc[:, 'month'] = df.index.month
    df.loc[:, 'day_in_month'] = df.index.day

    # prepare df for agg
    idx_name = df.index.name
    df = df.reset_index()

    # take care of leakage
    df_for_agg = df
    df_for_agg = df_for_agg[df_for_agg[idx_name] < test_start_date].copy()

    # group by calendar week and day of week
    df_agg_kw_day = df_for_agg.groupby(['year_week', 'week_day', 'ressort']).agg(
        agg_kw_day_anzahl_positionen_mean=('anzahl_positionen', "mean"),
        agg_kw_day_anzahl_positionen_max=('anzahl_positionen', "max"),
        agg_kw_day_anzahl_positionen_min=('anzahl_positionen', "min"),
        agg_kw_day_markting_cost_mean=('cost', "mean"),
        agg_kw_day_markting_cost_max=('cost', "max"),
        agg_kw_day_markting_cost_min=('cost', "min"),
        agg_kw_day_anzahl_bestellungen_mean=('anzahl_bestellungen', "mean"),
        agg_kw_day_anzahl_bestellungen_max=('anzahl_bestellungen', "max"),
        agg_kw_day_anzahl_bestellungen_min=('anzahl_bestellungen', "min"),
        agg_kw_day_avg_price_mean=('avg_price', 'mean'),
        agg_kw_day_avg_price_max=('avg_price', 'max'),
        agg_kw_day_avg_price_min=('avg_price', 'min'),
        # pricereduction features
        agg_kw_day_total_abs_reduk_mean=('total_abs_reduk', 'mean'),
        agg_kw_day_total_abs_reduk_max=('total_abs_reduk', 'max'),
        agg_kw_day_total_abs_reduk_min=('total_abs_reduk', 'min'),
        agg_kw_day_total_rel_reduk_mean=('total_rel_reduk', 'mean'),
        agg_kw_day_total_rel_reduk_max=('total_rel_reduk', 'max'),
        agg_kw_day_total_rel_reduk_min=('total_rel_reduk', 'min'),
        agg_kw_day_max_abs_reduk_mean=('max_abs_reduk', 'mean'),
        agg_kw_day_max_abs_reduk_max=('max_abs_reduk', 'max'),
        agg_kw_day_max_abs_reduk_min=('max_abs_reduk', 'min'),
        agg_kw_day_max_rel_reduk_mean=('max_rel_reduk', 'mean'),
        agg_kw_day_max_rel_reduk_max=('max_rel_reduk', 'max'),
        agg_kw_day_max_rel_reduk_min=('max_rel_reduk', 'min'),
        agg_kw_day_min_abs_reduk_mean=('min_abs_reduk', 'mean'),
        agg_kw_day_min_abs_reduk_max=('min_abs_reduk', 'max'),
        agg_kw_day_min_abs_reduk_min=('min_abs_reduk', 'min'),
        agg_kw_day_min_rel_reduk_max=('min_rel_reduk', 'max'),
        agg_kw_day_min_rel_reduk_mean=('min_rel_reduk', 'mean'),
        agg_kw_day_min_rel_reduk_min=('min_rel_reduk', 'min'),
        agg_kw_day_avg_abs_reduk_mean=('avg_abs_reduk', 'mean'),
        agg_kw_day_avg_abs_reduk_max=('avg_abs_reduk', 'max'),
        agg_kw_day_avg_abs_reduk_min=('avg_abs_reduk', 'min'),
        agg_kw_day_avg_rel_reduk_mean=('avg_rel_reduk', 'mean'),
        agg_kw_day_avg_rel_reduk_max=('avg_rel_reduk', 'max'),
        agg_kw_day_avg_rel_reduk_min=('avg_rel_reduk', 'min'),
        agg_kw_day_stddev_abs_reduk_mean=('stddev_abs_reduk', 'mean'),
        agg_kw_day_stddev_abs_reduk_max=('stddev_abs_reduk', 'max'),
        agg_kw_day_stddev_abs_reduk_min=('stddev_abs_reduk', 'min'),
        agg_kw_day_stddev_rel_reduk_mean=('stddev_rel_reduk', 'mean'),
        agg_kw_day_stddev_rel_reduk_max=('stddev_rel_reduk', 'max'),
        agg_kw_day_stddev_rel_reduk_min=('stddev_rel_reduk', 'min')
    )

    df_agg_kw_day = df_agg_kw_day.reset_index(level=['year_week', 'week_day', 'ressort'])

    # group by day of year
    df_agg_dayofyear = df_for_agg.groupby(['dayofyear', 'ressort']).agg(
        agg_dayofyear_anzahl_positionen_mean=('anzahl_positionen', "mean"),
        agg_dayofyear_anzahl_positionen_max=('anzahl_positionen', "max"),
        agg_dayofyear_anzahl_positionen_min=('anzahl_positionen', "min"),
        agg_dayofyear_markting_cost_mean=('cost', "mean"),
        agg_dayofyear_markting_cost_max=('cost', "max"),
        agg_dayofyear_markting_cost_min=('cost', "min"),
        agg_dayofyear_anzahl_bestellungen_mean=('anzahl_bestellungen', "mean"),
        agg_dayofyear_anzahl_bestellungen_max=('anzahl_bestellungen', "max"),
        agg_dayofyear_anzahl_bestellungen_min=('anzahl_bestellungen', "min"),
        agg_dayofyear_avg_price_mean=('avg_price', 'mean'),
        agg_dayofyear_avg_price_max=('avg_price', 'max'),
        agg_dayofyear_avg_price_min=('avg_price', 'min'),
        # pricereduction features
        agg_dayofyear_total_abs_reduk_mean=('total_abs_reduk', 'mean'),
        agg_dayofyear_total_abs_reduk_max=('total_abs_reduk', 'max'),
        agg_dayofyear_total_abs_reduk_min=('total_abs_reduk', 'min'),
        agg_dayofyear_total_rel_reduk_mean=('total_rel_reduk', 'mean'),
        agg_dayofyear_total_rel_reduk_max=('total_rel_reduk', 'max'),
        agg_dayofyear_total_rel_reduk_min=('total_rel_reduk', 'min'),
        agg_dayofyear_max_abs_reduk_mean=('max_abs_reduk', 'mean'),
        agg_dayofyear_max_abs_reduk_max=('max_abs_reduk', 'max'),
        agg_dayofyear_max_abs_reduk_min=('max_abs_reduk', 'min'),
        agg_dayofyear_max_rel_reduk_mean=('max_rel_reduk', 'mean'),
        agg_dayofyear_max_rel_reduk_max=('max_rel_reduk', 'max'),
        agg_dayofyear_max_rel_reduk_min=('max_rel_reduk', 'min'),
        agg_dayofyear_min_abs_reduk_mean=('min_abs_reduk', 'mean'),
        agg_dayofyear_min_abs_reduk_max=('min_abs_reduk', 'max'),
        agg_dayofyear_min_abs_reduk_min=('min_abs_reduk', 'min'),
        agg_dayofyear_min_rel_reduk_mean=('min_rel_reduk', 'mean'),
        agg_dayofyear_min_rel_reduk_max=('min_rel_reduk', 'max'),
        agg_dayofyear_min_rel_reduk_min=('min_rel_reduk', 'min'),
        agg_dayofyear_avg_abs_reduk_mean=('avg_abs_reduk', 'mean'),
        agg_dayofyear_avg_abs_reduk_max=('avg_abs_reduk', 'max'),
        agg_dayofyear_avg_abs_reduk_min=('avg_abs_reduk', 'min'),
        agg_dayofyear_avg_rel_reduk_mean=('avg_rel_reduk', 'mean'),
        agg_dayofyear_avg_rel_reduk_max=('avg_rel_reduk', 'max'),
        agg_dayofyear_avg_rel_reduk_min=('avg_rel_reduk', 'min'),
        agg_dayofyear_stddev_abs_reduk_mean=('stddev_abs_reduk', 'mean'),
        agg_dayofyear_stddev_abs_reduk_max=('stddev_abs_reduk', 'max'),
        agg_dayofyear_stddev_abs_reduk_min=('stddev_abs_reduk', 'min'),
        agg_dayofyear_stddev_rel_reduk_mean=('stddev_rel_reduk', 'mean'),
        agg_dayofyear_stddev_rel_reduk_max=('stddev_rel_reduk', 'max'),
        agg_dayofyear_stddev_rel_reduk_min=('stddev_rel_reduk', 'min')
    )
    df_agg_dayofyear = df_agg_dayofyear.reset_index(level=['dayofyear', 'ressort'])

    # group by weekday
    df_agg_weekday = df_for_agg.groupby(['week_day', 'ressort']).agg(
        agg_week_day_anzahl_positionen_mean=('anzahl_positionen', "mean"),
        agg_week_day_anzahl_positionen_max=('anzahl_positionen', "max"),
        agg_week_day_anzahl_positionen_min=('anzahl_positionen', "min"),
        agg_week_day_markting_cost_mean=('cost', "mean"),
        agg_week_day_markting_cost_max=('cost', "max"),
        agg_week_day_markting_cost_min=('cost', "min"),
        agg_week_day_anzahl_bestellungen_mean=('anzahl_bestellungen', "mean"),
        agg_week_day_anzahl_bestellungen_max=('anzahl_bestellungen', "max"),
        agg_week_day_anzahl_bestellungen_min=('anzahl_bestellungen', "min"),
        agg_week_day_avg_price_mean=('avg_price', 'mean'),
        agg_week_day_avg_price_max=('avg_price', 'max'),
        agg_week_day_avg_price_min=('avg_price', 'min'),
        agg_week_day_avg_retouren_mean=('anzahl_retouren', 'mean'),
        agg_week_day_avg_retouren_max=('anzahl_retouren', 'max'),
        # pricereduction features
        agg_week_day_total_rel_reduk_mean=('total_rel_reduk', 'mean'),
        agg_week_day_total_rel_reduk_max=('total_rel_reduk', 'max'),
        agg_week_day_total_rel_reduk_min=('total_rel_reduk', 'min'),
        agg_week_day_max_abs_reduk_mean=('max_abs_reduk', 'mean'),
        agg_week_day_max_abs_reduk_max=('max_abs_reduk', 'max'),
        agg_week_day_max_abs_reduk_min=('max_abs_reduk', 'min'),
        agg_week_day_max_rel_reduk_mean=('max_rel_reduk', 'mean'),
        agg_week_day_max_rel_reduk_max=('max_rel_reduk', 'max'),
        agg_week_day_max_rel_reduk_min=('max_rel_reduk', 'min'),
        agg_week_day_min_abs_reduk_mean=('min_abs_reduk', 'mean'),
        agg_week_day_min_abs_reduk_max=('min_abs_reduk', 'max'),
        agg_week_day_min_abs_reduk_min=('min_abs_reduk', 'min'),
        agg_week_day_min_rel_reduk_mean=('min_rel_reduk', 'mean'),
        agg_week_day_min_rel_reduk_max=('min_rel_reduk', 'max'),
        agg_week_day_min_rel_reduk_min=('min_rel_reduk', 'min'),
        agg_week_day_avg_abs_reduk_mean=('avg_abs_reduk', 'mean'),
        agg_week_day_avg_abs_reduk_max=('avg_abs_reduk', 'max'),
        agg_week_day_avg_abs_reduk_min=('avg_abs_reduk', 'min'),
        agg_week_day_avg_rel_reduk_mean=('avg_rel_reduk', 'mean'),
        agg_week_day_avg_rel_reduk_max=('avg_rel_reduk', 'max'),
        agg_week_day_avg_rel_reduk_min=('avg_rel_reduk', 'min'),
        agg_week_day_stddev_abs_reduk_mean=('stddev_abs_reduk', 'mean'),
        agg_week_day_stddev_abs_reduk_max=('stddev_abs_reduk', 'max'),
        agg_week_day_stddev_abs_reduk_min=('stddev_abs_reduk', 'min'),
        agg_week_day_stddev_rel_reduk_mean=('stddev_rel_reduk', 'mean'),
        agg_week_day_stddev_rel_reduk_max=('stddev_rel_reduk', 'max'),
        agg_week_day_stddev_rel_reduk_min=('stddev_rel_reduk', 'min')
    )
    df_agg_weekday = df_agg_weekday.reset_index(level=['week_day', 'ressort'])

    # create rolling averages
    no_of_days_for_rolling_avg = cfg['feature_engineering']['no_of_days_for_rolling_avg']
    all_ressorts = df_for_agg['ressort'].drop_duplicates()
    df_agg_rolling = df_for_agg[
        ['Datum', 'ressort', 'anzahl_positionen', 'anzahl_bestellungen', 'anzahl_retouren', 'avg_abs_reduk', 'avg_rel_reduk']].copy()
    for ressort in all_ressorts:
        filter_condition_ressort = df_agg_rolling.ressort == ressort
        df_agg_rolling.loc[filter_condition_ressort, 'agg_rolling_anzahl_positionen'] = \
        df_agg_rolling[filter_condition_ressort]['anzahl_positionen'].rolling(
            window=no_of_days_for_rolling_avg).mean()
        df_agg_rolling.loc[filter_condition_ressort, 'agg_rolling_anzahl_bestellungen'] = \
        df_agg_rolling[filter_condition_ressort]['anzahl_bestellungen'].rolling(
            window=no_of_days_for_rolling_avg).mean()
        df_agg_rolling.loc[filter_condition_ressort, 'agg_rolling_anzahl_retouren'] = \
        df_agg_rolling[filter_condition_ressort]['anzahl_retouren'].rolling(
            window=no_of_days_for_rolling_avg).mean()
        df_agg_rolling.loc[filter_condition_ressort, 'agg_rolling_avg_abs_reduk'] = \
        df_agg_rolling[filter_condition_ressort]['avg_abs_reduk'].rolling(
            window=no_of_days_for_rolling_avg).mean()
        df_agg_rolling.loc[filter_condition_ressort, 'agg_rolling_avg_rel_reduk'] = \
        df_agg_rolling[filter_condition_ressort]['avg_rel_reduk'].rolling(
            window=no_of_days_for_rolling_avg).mean()
    delta = test_end_date - test_start_date
    df_agg_rolling['Datum'] = df_agg_rolling['Datum'] + timedelta(days=delta.days)
    df_agg_rolling = df_agg_rolling.reset_index()

    # group by year and month for calculating a cumulative sum
    df_for_agg_cumu = df_for_agg[['year', 'month', 'day_in_month', 'ressort', 'cost']].copy()
    df_for_agg_cumu = df_for_agg_cumu.reset_index()
    df_for_agg_cumu = df_for_agg_cumu.set_index(['ressort', 'year', 'month', 'day_in_month'])
    df_for_agg_cumu = df_for_agg_cumu.sort_index()

    # TODO: Fix cumulated sum
    # df_for_agg_cumu = df_for_agg_cumu.fillna(0)
    # save_df_to_file(df_for_agg_cumu, 'aktion', type='pickle')
    # df_agg_cumu = df_for_agg_cumu.groupby(['ressort', 'year', 'month', 'day_in_month']).agg(
    #     agg_year_month_cost_cumsum=('cost', "cumsum")
    # )

    # df_agg_cumu = df_agg_cumu.reset_index()
    # df_agg_cumu[idx_name] = df_agg_cumu.apply(lambda x: date(int(x['year']), int(x['month']),
    #                                                          int(x['day_in_month'])), axis=1)
    # df_agg_cumu[idx_name] = df_agg_cumu[idx_name].astype('datetime64')
    # df_agg_cumu = df_agg_cumu[[idx_name, 'agg_year_month_cost_cumsum']]

    # merge agg dfs onto original df
    df = df.merge(df_agg_kw_day, on=['year_week', 'week_day', 'ressort'], how='left')
    df = df.merge(df_agg_dayofyear, on=['dayofyear', 'ressort'], how='left')
    df = df.merge(df_agg_weekday, on=['week_day', 'ressort'], how='left')
    # df = df.merge(df_agg_cumu, on=[idx_name, 'ressort'], how='left')
    df = df.merge(df_agg_rolling[
                      ['Datum', 'ressort', 'agg_rolling_anzahl_positionen', 'agg_rolling_anzahl_bestellungen',
                       'agg_rolling_anzahl_retouren', 'agg_rolling_avg_abs_reduk', 'agg_rolling_avg_rel_reduk']],
                  on=['Datum', 'ressort'], how='left')
    # df = df.fillna(0)

    # set index
    df = df.set_index(idx_name)

    df = df[df.columns.difference(drop_col)].copy()

    # save output for debugging
    # save_df_to_bigquery(df, project, dataset_name, feature_importance_table+'_avg', write_disposition='WRITE_TRUNCATE')

    return df


def build_saison_features(df):
    """
        In dieser Funktion werden Features generiert, die die Saisonalität beschreiben.

        :param p_df: Dataframe, das um die Saison Features erweitert werden soll.
        :return: pandas.Dataframe: Gibt den als Input gegebenen Dataframe zurück, aber um weitere Saison Features erweitert.
    """
    logger.debug("Entered build_saison_features()")

    idx_name = df.index.name
    df = df.reset_index()

    df['month_day'] = df['full_date'].dt.strftime('%m-%d')
    df['saison_FS'] = np.where((df['month_day'] >= '03-01') & (df['month_day'] < '09-01'), 1, 0)
    df['saison_HW'] = np.where((df['month_day'] >= '09-01') | (df['month_day'] < '03-01'), 1, 0)

    df = df.drop('month_day', axis=1)
    df = df.set_index(idx_name)

    # save output for debugging
    # save_df_to_bigquery(df, project, dataset_name, feature_importance_table+'_saison', write_disposition='WRITE_TRUNCATE')

    return df
