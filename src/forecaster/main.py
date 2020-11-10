import yaml
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
import src.forecaster.logs as logs
import src.forecaster.utils as utils
import src.forecaster.configs_for_code as cfg
import src.forecaster.prepare_data as prep
import src.forecaster.feat_engineering as feat
import src.forecaster.model as model
import src.forecaster.output as output

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)

def main():
    """
        Main function, executes all pipelines:
            data ingest,
            data preparation,
            feature engineering,
            model,
            post processing,
            output
    """

    df_all_data = prep.load_data()

    df_with_features = feat.generate_features(df_all_data)

    df_with_preds = model.generate_predictions(df_with_features)

    output.export_results(df_with_preds)


if __name__ == "__main__":
    main()
