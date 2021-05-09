# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import sys
import Azure_ML.aml_pipeline_for_durable as pipeline


def main(pipelineName) -> str:
    print(f"Triggering pipeline endpoint '{pipelineName}' (as configured)...")
    msg, pipeline_run = pipeline.start_sonntagsfrage_pipeline()

    return f"Pipeline {pipelineName} has been successfully started!" #, pipeline_run
