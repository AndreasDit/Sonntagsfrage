import yaml
import os
import sys
import logging
import azure.functions as func

sys.path.append(os.getcwd())
import utils.logs as logs
import utils.configs_for_code as cfg
import forecaster.main as forecast_main

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)
logger = logs.create_logger(__name__)


def main(req: func.HttpRequest) -> func.HttpResponse:
    """    This Azure Function calls the forecaster and performs the forecasting of the next value.
    """
    logging.info('Python HTTP trigger function starts producing a forecast.')

    forecast_main.main()

    return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
