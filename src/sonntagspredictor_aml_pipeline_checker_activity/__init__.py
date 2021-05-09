# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import Azure_ML.aml_pipeline_for_durable as pipeline


def main(pipelineRun) -> str:
    # --- wait for started pipeline to finish
    print(f"Waiting for pipeline  to finish...")
    result = pipeline.check_sonntagsfrage_pipeline(pipelineRun)

    return f"Pipeline has finished successfully!"
