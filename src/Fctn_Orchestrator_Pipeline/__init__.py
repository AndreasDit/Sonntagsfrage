# This function is not intended to be invoked directly. Instead it will be
# triggered by an HTTP starter function.
# Before running this sample, please:
# - create a Durable activity function (default name is "Hello")
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import json

import azure.functions as func
import azure.durable_functions as df


def orchestrator_function(context: df.DurableOrchestrationContext):
    result1 = yield context.call_activity('crawl_data_questionaire_results_activity')
    result2 = yield context.call_activity('clean_crawled_data_activity')
    result3 = yield context.call_activity('forecaster_activity')
    result4 = yield context.call_activity('google_spreadsheets_activity')
    return [
        result1, 
        result2, result3, result4]

main = df.Orchestrator.create(orchestrator_function)