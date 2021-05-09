import logging
import Azure_ML.aml_pipeline_for_durable as pipeline

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    print(f"Waiting for pipeline  to finish...")
    msg, pipiline_run = pipeline.start_sonntagsfrage_pipeline()

    return func.HttpResponse(
        "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
        status_code=200
    )
