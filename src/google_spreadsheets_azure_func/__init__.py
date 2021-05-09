import logging
import azure.functions as func
import google_spreadsheets.load_data_spreadsheet as google


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python trigger function loaded date into google spreadsheets.')

    google.main()

    return func.HttpResponse(
        json.dumps({
        'response':"This HTTP triggered function did load data into a google spreadsheet."
        }),
        status_code=200
    )
