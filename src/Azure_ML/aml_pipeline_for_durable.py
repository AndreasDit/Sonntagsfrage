# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import sys
import json
from azureml.core import Experiment, Workspace, Environment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineEndpoint
from azureml.core.authentication import ServicePrincipalAuthentication
from environs import Env


def start_sonntagsfrage_pipeline():
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
    workspaceName = env("WORKSPACE_NAME")

    # --- get creds for aservice principalv
    with open('Azure_ML/service_principals/sonntagsfrage-ml-auth-file.json') as f:
        svcpr_data = json.load(f)

    # --- get service principal
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=svcpr_data['tenantId'],
        service_principal_id=svcpr_data['clientId'],
        service_principal_password=svcpr_data['clientSecret'])

    # --- get workspace, compute target, run config
    print("Getting workspace and compute target...")
    workspace = Workspace(
        subscription_id=azure_subscription_id,
        resource_group=resource_group,
        workspace_name=workspaceName,
        auth=svc_pr
    )

    print(f"Get pipeline endpoint '{pipelineName}' (as configured)...")
    pipeline_endpoint = PipelineEndpoint.get(
        workspace=workspace, name=pipelineName)

    print(f"Triggering pipeline endpoint '{pipelineName}' (as configured)...")
    pipeline_run = Experiment(
        workspace, pipelineName).submit(pipeline_endpoint)

    return f"Pipeline {pipelineName} has been successfully started!", pipeline_run


def check_sonntagsfrage_pipeline(pipelineRun) -> str:
    # --- wait for started pipeline to finish
    print(f"Waiting for pipeline  to finish...")
    pipelineRun.wait_for_completion()

    return f"Pipeline has finished successfully!"


msg, pipiline_run = start_sonntagsfrage_pipeline()

print(
    f"Start checking for pipeline status of pipeline run '{pipiline_run}' ...")
result = check_sonntagsfrage_pipeline(pipiline_run)
print(result)
