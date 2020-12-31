import pickle
from datetime import datetime

import yaml

import code_configs
import dataprep as dp
import feature_engineering as feat
import log
import model
import post_processing as post
from helper import dynamic_train_test_date

file = open(code_configs.CONFIG_PATH, 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

logger = log.get_logger(__name__)


def main():
    """
        Main Funktion dieser Pipeline. Einstiegspunkt für das Ausführen der Pipeline.

        Führt nacheinander die Steps der Pipeline durch. Im Wesentlichen:
            * Data Input
            * Data Prep
            * Feature Engineering
            * Model Training
            * Output
    """
    logger.debug("Entered main()")
    model_path = cfg['main']['model_path']
    project = cfg['data_input']['project_name']
    dataset_name = cfg['data_input']['dataset_name']
    bucket_name = cfg['data_input']['bucket_name']
    result_file_path = cfg['main']['result_file_path']
    result_file_name = cfg['main']['result_file_name']
    result_bigquery_table = cfg['main']['result_bigquery_table']
    export_columns = cfg['main']['export_columns']
    bigquery_date_col = cfg['data_input']['bigquery_date_col']
    train_mode = cfg['model']['train_mode']
    pred_mode = cfg['main']['pred_mode']
    feature_save_mode = cfg['main']['feature_save_mode']
    feature_bigquery_table = cfg['main']['feature_bigquery_table']
    feature_prediction_bigquery_table = cfg['main']['feature_prediction_bigquery_table']

    time = datetime.today()
    result_file_name = result_file_name.replace('<YYYYmmddHMS>', time.strftime('%Y%m%d%H%M%S'))

    logger.info("Started Input, Prep and Feat. Engineering")

    # load all data
    df_input = dp.data_input()

    df = dp.data_prep(df_input)

    df = feat.feature_engineering(df)
    dp.save_df_to_file(df, filename='df_input_post_feat', type='pickle')
    if feature_save_mode == 'bigquery':
        dp.save_df_to_bigquery(df, project, dataset_name, feature_bigquery_table, write_disposition='WRITE_TRUNCATE')
    
    logger.info("Model training")
    train_start_date = cfg['model']['train_start_date']
    start_date, end_date = dynamic_train_test_date()

    train, test = model.date_splitter(df, train_start_date, start_date, end_date)

    # save DFs for further inspection
    dp.save_df_to_file(train, filename='train', type='pickle')
    dp.save_df_to_file(test, filename='test', type='pickle')

    df, trained_model = model.calc_preds(train, test, train_mode, pred_mode)
    # pickle.dump(trained_model, open(model_path, "wb"))
    logger.info("Model training done")

    # save DFs for further inspection
    dp.save_df_to_file(df, filename='df_with_pred', type='pickle')
    dp.save_df_to_bigquery(df, project, dataset_name, feature_prediction_bigquery_table, write_disposition='WRITE_TRUNCATE')

    logger.info("Starting post processing")
    df = post.process(df, export_columns)

    logger.info("exporting data")
    dp.save_df_to_file(df, filename='df_input', type='pickle')
    dp.save_df_to_gcs(df, result_file_name, bucket_name, result_file_path, type='csv')
    dp.save_df_to_bigquery(df, project, dataset_name, result_bigquery_table)

    logger.info("Done!")


if __name__ == "__main__":
    main()
