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
    pipeline_run_from_starter = None
    aml_pipeline_starter = {"pipeline_name": "Sonntagsfrage-Forecaster-Pipeline",
                            "workspace_name": "Sonntagsfrage-predictor"}

    # --- start data scraping
    result1 = yield context.call_activity('crawl_data_questionaire_results_activity')

    # --- start data cleaning
    result2 = yield context.call_activity('clean_crawled_data_activity')

    # --- start predicting values via the Azure ML Service
    result3, pipeline_run_from_starter = yield context.call_activity('sonntagspredictor_aml_pipeline_starter_activity', aml_pipeline_starter)
    deadline = context.current_utc_datetime + timedelta(minutes=30)
    yield context.create_timer(deadline)
    aml_pipeline_checker = {"pipeline_name": "Sonntagsfrage-Forecaster-Pipeline",
                            "workspace_name": "Sonntagsfrage-predictor",
                            "pipeline_run": pipeline_run_from_starter}
    result4 = yield context.call_activity('sonntagspredictor_aml_pipeline_starter_activity', aml_pipeline_checker)

    # --- start transferring data to the google sheet service
    result5 = yield context.call_activity('google_spreadsheets_activity')

    return [
        result1,
        result2, result3, result4, result5]


main = df.Orchestrator.create(orchestrator_function)
