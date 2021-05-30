"""Main script for step: Send Data to Google spreadsheet"""

import os
import argparse
from azureml.core import Run
import google_spreadsheets.load_data_spreadsheet as google


print("Sending data...")


# --- initialization
print("Initialization...")

# - define and parse script arguments
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--input-dir", type=str, required=True, help="input directory")
parser.add_argument("--output-dir", type=str, required=True, help="output directory")
parser.add_argument("--g-type", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--project-id", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--private-key-id", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--private-key", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--client-email", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--client-id", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--auth-uri", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--token-uri", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--auth-provider-x509-cert-url", type=str, required=True, help="parameter for google credentials")
parser.add_argument("--client-x509-cert-url", type=str, required=True, help="parameter for google credentials")
args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
g_type = args.g_type
project_id = args.project_id
private_key_id = args.private_key_id
private_key = args.private_key
client_email = args.client_email
client_id = args.client_id
auth_uri = args.auth_uri
token_uri = args.token_uri
auth_provider_x509_cert_url = args.auth_provider_x509_cert_url
client_x509_cert_url = args.client_x509_cert_url

google_credentials = {
    "type":  g_type ,
    "project_id":  project_id ,
    "private_key_id":  private_key_id ,
    "private_key":  private_key ,
    "client_email":  client_email ,
    "client_id":  client_id ,
    "auth_uri":  auth_uri ,
    "token_uri":  token_uri ,
    "auth_provider_x509_cert_url":  auth_provider_x509_cert_url ,
    "client_x509_cert_url":  client_x509_cert_url 
}

# - get run context
run = Run.get_context()

# - ensure that the output directory exists
print("Ensuring that the output directory exists...")


print("cleaning data from Azure SQL DB ...")
google.main(google_credentials)

# --- Done
print("Done.")