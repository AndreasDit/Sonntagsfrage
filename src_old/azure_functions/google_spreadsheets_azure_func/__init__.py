import logging
import azure.functions as func
import google_spreadsheets.load_data_spreadsheet as google


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python trigger function loaded date into google spreadsheets.')

    google.main()

    return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
    )
