# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging


def main(pipeline_name: str) -> str:
    print(f"Triggering pipeline endpoint '{pipeline_name}' (as configured)...")
    pipeline_run = Experiment(
        workspace, pipeline_name).submit(pipeline_endpoint)
    pipeline_run.wait_for_completion()
    return f"Pipeline {pipeline_name} completed successfully!"
