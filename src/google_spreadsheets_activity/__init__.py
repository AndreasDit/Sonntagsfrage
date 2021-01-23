# This function is not intended to be invoked directly. Instead it will be
# triggered by an orchestrator function.
# Before running this sample, please:
# - create a Durable orchestration function
# - create a Durable HTTP starter function
# - add azure-functions-durable to requirements.txt
# - run pip install -r requirements.txt

import logging
import google_spreadsheets.load_data_spreadsheet as google


def main(name: str) -> str:

    google.main()

    return f"Data was successfully loaded into google spreadsheets."
