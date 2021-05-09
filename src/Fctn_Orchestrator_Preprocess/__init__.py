# This function is not intended to be invoked directly. Instead it will be
# triggered by an HTTP starter function.
# Before running this sample, please:
# - create a Durable activity function (default name is "Hello")
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import json
from datetime import datetime, timedelta
import azure.functions as func
import azure.durable_functions as df


def orchestrator_function(context: df.DurableOrchestrationContext):
    # --- define parameters
    aml_pipeline_starter = {"pipeline_name": "Sonntagsfrage-Forecaster-Pipeline",
                            "workspace_name": "Sonntagsfrage-predictor"}

    # --- start data scraping
    result1 = yield context.call_activity('crawl_data_questionaire_results_activity')

    # --- start data cleaning
    result2 = yield context.call_activity('clean_crawled_data_activity')

    # --- start predicting values via the Azure ML Service
    result3 = yield context.call_activity('sonntagspredictor_aml_pipeline_starter_activity', aml_pipeline_starter)

    return [
        result1,
        result2
        , result3
        ]


main = df.Orchestrator.create(orchestrator_function)
