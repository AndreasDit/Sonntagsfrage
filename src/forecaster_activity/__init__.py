# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import forecaster.main as forecast_main

def main(name: str) -> str:
    """    This Activity calls the forecaster and performs the forecasting of the next value.
    """
    logging.info('Activity function starts producing a forecast.')

    forecast_main.main()

    return "The forecast was calculated successfully by this activity function."
