# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import sys
from azureml.core import Experiment, Workspace, Environment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineEndpoint
from environs import Env


def main(pipelineRun) -> str:
    # --- wait for started pipeline to finish
    print(f"Waiting for pipeline  to finish...")
    pipelineRun.wait_for_completion()

    return f"Pipeline has finished successfully!"
