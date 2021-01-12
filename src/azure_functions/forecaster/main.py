import yaml
import os
import sys

sys.path.append(os.getcwd())
print(os.getcwd())
sys.path.append('/Users/andreasditte/Desktop/Private_Projekte/Sonntagsfrage/src/azure_functions/')
import utils.logs as logs
import utils.configs_for_code as cfg
import forecaster.prepare_data as prep
import forecaster.feat_engineering as feat
import forecaster.model as model
import forecaster.output as output

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

    df_with_preds, df_metrics = model.combined_restults_from_all_algorithms(df_with_features)

    output.export_results(df_with_preds)
    output.export_metrics(df_metrics)


if __name__ == "__main__":
    main()
