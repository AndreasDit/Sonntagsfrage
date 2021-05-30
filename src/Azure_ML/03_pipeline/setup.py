"""Creates and deploys the model training and deployment pipeline."""

# pylint: disable=unused-import
import argparse
import sys
from pathlib import Path

from azureml.core import Experiment, Workspace, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineEndpoint
from azureml.pipeline.core.schedule import Schedule, ScheduleRecurrence
from azureml.pipeline.steps import EstimatorStep, HyperDriveStep, PythonScriptStep
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import (
    BanditPolicy,
    BayesianParameterSampling,
    HyperDriveConfig,
    PrimaryMetricGoal,
    RandomParameterSampling,
    choice,
)
from azureml.core.authentication import ServicePrincipalAuthentication
from environs import Env
import json

# pylint: enable=unused-import

# --- define and parse script arguments

# --- load configuration
print("Loading configuration...")
for p in sys.path:
    print(p)
env = Env()
env.read_env("Azure_ML/foundation.env")

azure_subscription_id = env("AZURE_SUBSCRIPTION_ID")
resource_group = env("RESOURCE_GROUP")
workspace_name = env("WORKSPACE_NAME")
workspace_region = env("WORKSPACE_REGION")
gpu_cluster_name = env("GPU_BATCH_CLUSTER_NAME")
cpu_cluster_name = env("CPU_BATCH_CLUSTER_NAME")
schedule = env("SCHEDULE")
trigger_after_publish = env("TRIGGER_AFTER_PUBLISH")
sql_name = env("SQL_DB_NAME")
sql_pw = env("SQL_DB_PW")

# --- Google Service Account
g_type = env("TYPE")
project_id = env("PROJECT_ID")
private_key_id = env("PRIVATE_KEY_ID")
private_key = env("PRIVATE_KEY")
client_email = env("CLIENT_EMAIL")
client_id = env("CLIENT_ID")
auth_uri = env("AUTH_URI")
token_uri = env("TOKEN_URI")
auth_provider_x509_cert_url = env("AUTH_PROVIDER_X509_CERT_URL")
client_x509_cert_url = env("CLIENT_X509_CERT_URL")



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
    workspace_name=workspace_name,
    auth=svc_pr
)

compute_target = ComputeTarget(workspace=workspace, name=cpu_cluster_name)

requirements_path = Path("./requirements.txt").resolve()
environment = Environment.from_pip_requirements(
    name="SpacyEnvironment", file_path=requirements_path
)

run_config = RunConfiguration()
run_config.environment = environment
# run_config = RunConfiguration(
#     conda_dependencies=CondaDependencies.create(
#         conda_packages=['pip', 'pandas', 'scikit-learn', 'PyYAML'],
#         # notes: - see run_config.environment.add_private_pip_wheel() to use your own private packages,
#         #        - you can also reference curated or custom environments here for simplification,
#         #          see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline for
#         #          more details
#         pip_packages=["azureml-defaults", "azureml-pipeline-steps"],
#     )
# )


# define if docker images should be created or not
# run_config.environment.docker.enabled = False
run_config.environment.docker.enabled = True
# run_config.environment.docker.base_image = 'aksanakuzmitskaya/azml_pyodbc:secondtry'
run_config.environment.docker.base_image = 'adtest123/azureml-pyodbc-conda:latest'


# recommendation: use a fixed image in production to avoid sudden surprises
#                 check DEFAULT_CPU_IMAGE or DEFAULT_GPU_IMAGE for the newest image
# from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE
# run_config.environment.docker.base_image = (
# "mcr.microsoft.com/azureml/intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04:20200821.v1"
# )
run_config.environment.spark.precache_packages = False

# --- define pipeline steps
# note: use <pipline step>.run_after() if there is a dependency but no input/output connection
#       without these relations set up, all steps will run in parallel by default
print("Defining pipeline steps...")
pipeline_path = "Azure_ML/03_pipeline/"

# - Crawl Data
crawl_data_dir = PipelineData(
    "extracted_data",
    is_directory=True,
)
crawl_data_step = PythonScriptStep(
    name="Crawl Data",
    script_name=pipeline_path + "01_crawl_data/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    outputs=[crawl_data_dir],
    arguments=["--output-dir", crawl_data_dir,
               "--sql-name-in", sql_name,
               "--sql-pw-in", sql_pw,
               ],
    allow_reuse=False,
)

# - Clean Crawled Data
clean_data_dir = PipelineData(
    "extracted_data",
    is_directory=True,
)
clean_data_step = PythonScriptStep(
    name="Clean Crawled Data",
    script_name=pipeline_path + "02_clean_crawled_data/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[crawl_data_dir],
    outputs=[clean_data_dir],
    arguments=["--input-dir", crawl_data_dir,
               "--output-dir", clean_data_dir,
               "--sql-name-in", sql_name,
               "--sql-pw-in", sql_pw,
               ],
    allow_reuse=False,
)


# - Extract Data
extracted_data_dir = PipelineData(
    "extracted_data",
    is_directory=True,
)
extract_data_step = PythonScriptStep(
    name="Extract Data",
    script_name=pipeline_path + "03_extract_data/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[clean_data_dir],
    outputs=[extracted_data_dir],
    arguments=["--input-dir", clean_data_dir,
               "--output-dir", extracted_data_dir],
    allow_reuse=False,
)

# - Transform Data
transformed_data_dir = PipelineData(
    name="transformed_data",
    is_directory=True,
)
transform_data_step = PythonScriptStep(
    name="Transform Data",
    script_name=pipeline_path + "04_transform_data/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[extracted_data_dir],
    outputs=[transformed_data_dir],
    arguments=["--input-dir", extracted_data_dir,
               "--output-dir", transformed_data_dir],
    allow_reuse=False,
)

# - Calc Predfictions
calc_predictions_dir = PipelineData(
    name="calc_predictions",
    is_directory=True,
)
calc_predictions_step = PythonScriptStep(
    name="Calc Predictions",
    script_name=pipeline_path + "05_predict_data/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[transformed_data_dir],
    outputs=[calc_predictions_dir],
    arguments=["--input-dir", transformed_data_dir,
               "--output-dir", calc_predictions_dir],
    allow_reuse=False,
)

# - Send Data to Google spreadsheet
send_data_dir = PipelineData(
    "extracted_data",
    is_directory=True,
)
send_data_step = PythonScriptStep(
    name="Send Data to spreadsheet",
    script_name=pipeline_path + "06_send_data_to_googlesheet/main.py",
    source_directory='.',
    compute_target=compute_target,
    runconfig=run_config,
    inputs=[calc_predictions_dir],
    outputs=[send_data_dir],
    arguments=["--input-dir", calc_predictions_dir,
               "--output-dir", send_data_dir,
               "--g-type", g_type,
               "--project-id", project_id,
               "--private-key-id", private_key_id,
               "--private-key", private_key,
               "--client-email", client_email,
               "--client-id", client_id,
               "--auth-uri", auth_uri,
               "--token-uri", token_uri,
               "--auth-provider-x509-cert-url", auth_provider_x509_cert_url,
               "--client-x509-cert-url", client_x509_cert_url,
               ],
    allow_reuse=False,
)



# # - Train Models

# # # option 1 - no hyperparameter optimization, single model without hyperparameter tuning
# # more infos on estimators at:
# # https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.estimator.estimator.
# #
# # The example below uses the pre-defined TensorFlow estimator but there is also other estimators and the option to build
# # your own estimator.
# #
# # train_model_step = EstimatorStep(
# #     name="Train Model",
# #     estimator=TensorFlow(
# #         entry_script="main.py",
# #         source_directory="03_train_models",
# #         compute_target=compute_target,
# #         framework_version="2.0",
# #         conda_packages=[],
# #         pip_packages=["matplotlib"],
# #         use_gpu=True,
# #     ),
# #     compute_target=compute_target,
# #     inputs=[transformed_data_dir],
# #     estimator_entry_script_arguments=["--input-dir", transformed_data_dir],
# #     allow_reuse=False,
# # )

# # option 2 - hyperparameter optimization using HyperDrive
# # - see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters for more details and further
# #   options wrt. hyperparameter selection strategies and termination policies.
# # - also check if the settings below are valid in case you use this in a production context
# train_models_step = HyperDriveStep(
#     name="Train Models",
#     hyperdrive_config=HyperDriveConfig(
#         estimator=TensorFlow(
#             entry_script="main.py",
#             source_directory="03_train_models",
#             compute_target=compute_target,
#             framework_version="2.2",
#             conda_packages=[],
#             pip_packages=["matplotlib"],
#             use_gpu=True,
#         ),
#         hyperparameter_sampling=RandomParameterSampling(
#             {
#                 "--epochs": choice(10, 25, 50, 100),
#                 "--hidden-neurons": choice(10, 50, 200, 300, 500),
#                 "--batch-size": choice(32, 64, 128, 256),
#             }
#         ),
#         policy=BanditPolicy(evaluation_interval=3, slack_amount=0.05),
#         primary_metric_name="accuracy",
#         primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
#         max_total_runs=20,
#         max_concurrent_runs=4,
#         max_duration_minutes=120,
#     ),
#     inputs=[transformed_data_dir],
#     estimator_entry_script_arguments=["--input-dir", transformed_data_dir],
# )

# # - Register Best Model
# register_best_model_step = PythonScriptStep(
#     name="Register Best Model",
#     script_name="main.py",
#     source_directory="04_register_best_model",
#     compute_target=compute_target,
#     runconfig=run_config,
#     allow_reuse=False,
# )
# register_best_model_step.run_after(train_models_step)

# # - Deploy New Model
# deploy_new_model_step = PythonScriptStep(
#     name="Deploy New Model",
#     script_name="main.py",
#     source_directory="05_deploy_new_model",
#     compute_target=compute_target,
#     runconfig=run_config,
#     allow_reuse=False,
# )
# deploy_new_model_step.run_after(register_best_model_step)

# # --- assemble and publish publishing pipeline
# # note: see here for infos on how to schedule the pipeline
# #       https://github.com/Azure/MachineLearningNotebooks/blob/fe8fcd4b480dab7fee9fa32a354132e4df25db8a/how-to-use-azure
# #       ml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-setup-schedule-for-a-published-pipeline.ipynb
print("Assembling and publishing pipeline...")
pipeline_name = "Sonntagsfrage-Forecaster-Pipeline"
pipeline_description = "WiP pipeline."
pipeline_steps = [
    crawl_data_step,
    clean_data_step,
    extract_data_step,
    transform_data_step,
    calc_predictions_step,
    send_data_step,
    # trigger_pipeline_step,
    # train_models_step,
    # register_best_model_step,
    # deploy_new_model_step,
]
pipeline = Pipeline(
    workspace=workspace,
    steps=pipeline_steps,
    description=pipeline_description,
)
pipeline.validate()

published_pipeline = pipeline.publish(
    # name=pipeline_name, description=pipeline_description, version={...some version...}
    name=pipeline_name,
    description=pipeline_description,
)
print(f"Newly published pipeline id: {published_pipeline.id}")

try:
    pipeline_endpoint = PipelineEndpoint.get(
        workspace=workspace, name=pipeline_name)
    pipeline_endpoint.add(published_pipeline)
except:
    pipeline_endpoint = PipelineEndpoint.publish(
        workspace=workspace,
        name=pipeline_name,
        pipeline=published_pipeline,
        description=f"Pipeline Endpoint for {pipeline_name}",
    )

# TODO: cleanup older pipeline endpoints(?)


# --- add a schedule for the pipeline (if told to do so)
# note: this is a sample schedule which runs time-based.
#       there is also the option to trigger the pipeline based on changes.
#       details at https://github.com/Azure/MachineLearningNotebooks/blob/4e7b3784d50e81c313c62bcdf9a330194153d9cd/how-t
#       o-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-setup-schedule-for-a-published-pipelin
#       e.ipynb
if schedule:
    recurrence = ScheduleRecurrence(
        frequency="Day", interval=2, hours=[22], minutes=[30])
    schedule = Schedule.create(
        workspace=workspace,
        name="Every-Other-Day-At-10-30-PM",
        pipeline_id=published_pipeline.id,
        experiment_name=pipeline_name,
        recurrence=recurrence,
        wait_for_provisioning=True,
        description="A sample schedule which runs every other day at 10:30pm.",
    )


# --- trigger pipeline endpoint if we have been told to do so
if trigger_after_publish == True:
    print(f"Triggering pipeline endpoint '{pipeline_name}' (as configured)...")
    pipeline_run = Experiment(
        workspace, pipeline_name).submit(pipeline_endpoint)
    # pipeline_run.wait_for_completion()


# --- Done
print("Done.")
