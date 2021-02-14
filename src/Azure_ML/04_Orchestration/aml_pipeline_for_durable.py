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


def main():
    # --- read params
    pipelineName = "Sonntagsfrage-Forecaster-Pipeline"
    workspaceName = "Sonntagsfrage-predictor"

    # --- load configuration
    print("Loading configuration...")
    for p in sys.path:
        print(p)
    env = Env()
    env.read_env("Azure_ML/foundation.env")

    azure_subscription_id = env("AZURE_SUBSCRIPTION_ID")
    resource_group = env("RESOURCE_GROUP")
    workspaceName = env("workspace_Name")

    # --- get workspace, compute target, run config
    print("Getting workspace and compute target...")
    workspace = Workspace(
        subscription_id=azure_subscription_id,
        resource_group=resource_group,
        workspaceName=workspaceName,
    )

    print(f"Get pipeline endpoint '{pipelineName}' (as configured)...")
    pipeline_endpoint = PipelineEndpoint.get(
        workspace=workspace, name=pipelineName)

    print(f"Triggering pipeline endpoint '{pipelineName}' (as configured)...")
    pipeline_run = Experiment(
        workspace, pipelineName).submit(pipeline_endpoint)

    return f"Pipeline {pipelineName} has been successfully started!", pipeline_run


main()
