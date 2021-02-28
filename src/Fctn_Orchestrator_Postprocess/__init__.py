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
    # --- start transferring data to the google sheet service
    result5 = yield context.call_activity('google_spreadsheets_activity')

    return [result5]


main = df.Orchestrator.create(orchestrator_function)
