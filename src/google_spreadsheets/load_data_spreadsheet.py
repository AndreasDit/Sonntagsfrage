import gspread
import yaml
import os
import sys
sys.path.append(os.getcwd())

import src.configs_for_code as cfg

configs_file = open(cfg.PATH_CONFIG_FILE, 'r')
configs = yaml.load(configs_file, Loader=yaml.FullLoader)

WORKSHEET_NAME = configs['google']['worksheet_name']
PATH_GOOGLE_SERVICE_ACCOUNT = cfg.PATH_GOOGLE_SERVICE_ACCOUNT

# use creds to create a client to interact with the Google Drive API
scope = [
    'https://spreadsheets.google.com/feeds'
    # 'https://www.googleapis.com/auth/spreadsheets.readonly'
    # 'https://www.googleapis.com/auth/spreadsheets'
    # , 'https://www.googleapis.com/auth/drive'
         ]
# creds = ServiceAccountCredentials.from_json_keyfile_name('sonntagsfrage-295521-484072c2bc32.json', scope)
# client = gspread.authorize(creds)
gc = gspread.service_account(filename=PATH_GOOGLE_SERVICE_ACCOUNT)

sh = gc.open("Sonntagsfrage_data")
# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
# sheet = client.open("Sonntagsfrage_data").sheet1
# sheet = gc.open("Sonntagsfrage_data").sheet1

# manipulate sheet
# sheet.update_cell(1, 1, "I just wrote to a spreadsheet using Python!")

# worksheet_list = sh.worksheets()
# print(worksheet_list, worksheet_list[0].title)

worksheet = sh.worksheet(WORKSHEET_NAME)
# sh.del_worksheet(worksheet)

# worksheet = sh.add_worksheet(title=WORKSHEET_NAME, rows="100", cols="20")
# worksheet = sh.worksheet(WORKSHEET_NAME)
# sheet.delete_rows(start_index=1, end_index= nb_rows) # index starts at 1 and not at 0

row = []
index = 1
worksheet.insert_row(row, index)

nb_rows = worksheet.row_count
nb_rows_sjon = worksheet.get_all_records()
print(nb_rows, len(nb_rows_sjon))

worksheet.delete_rows(start_index=2, end_index=nb_rows+1)  # index starts at 1 and not at 0

row = ["I'm","inserting","a","row","into","a,","Spreadsheet","with","Python"]
index = 1
worksheet.insert_row(row, index)
row = [9,7,1,2,3,4,5,6,8]
index = 2
worksheet.insert_row(row, index)
row = [9,7,1,2,3,4,5,6,8]
index = 3
worksheet.insert_row(row, index)
row = [9,7,1,2,3,4,5,6,8]
index = 4
worksheet.insert_row(row, index) # insert puts new values at the top of the file


# Extract and print all of the values
list_of_hashes = worksheet.get_all_records()
print(list_of_hashes)
# print(sh.sheet1.get('A1'))
