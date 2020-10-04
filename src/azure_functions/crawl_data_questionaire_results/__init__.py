import logging
import pyodbc
import datetime as dt
import os
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    sql_name = os.getenv('sql_name')
    sql_pw = os.getenv('sql_pw')

    server = 'sonntagsfrage-server.database.windows.net'
    database = 'sonntagsfrage-sql-db'
    username = sql_name
    password = sql_pw 
    driver = '{ODBC Driver 17 for SQL Server}'

    # open connection
    conn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
    cursor = conn.cursor()

    # Do the insert
    sqlstmt = f"""insert into sonntagsfrage.test_poc(
        test, bla) values (
        'p', """+ str(5) +"""
        )"""
    cursor.execute(sqlstmt)
    # cursor.execute("""insert into sonntagsfrage.test_poc(
    #     test, bla, datum, ts) values (
    #     'pyodbc', 'awesome library', """ + dt.datetime.date() + """, """ dt.datetime.now() """
    #     )""")

    #commit the transaction
    # conn.commit()

    # insert rows into Azure SQL DB
    # for row in result:
    #     insertSql = "insert into TableName (Col1, Col2, Col3) values (?, ?, ?)"
    #     cursor.execute(insertSql, row[0], row[1], row[2])
    #     cursor.commit()
        
    # snippet for selecting data from azure sql
    # row = cursor.fetchone()
    # while row:
    #     print (str(row[0]) + " " + str(row[1]))
    #     row = cursor.fetchone()

    #     MERGE Production.UnitMeasure AS target  
    # USING (SELECT @UnitMeasureCode, @Name) AS source (UnitMeasureCode, Name)  
    # ON (target.UnitMeasureCode = source.UnitMeasureCode)  
    # WHEN MATCHED THEN
    #     UPDATE SET Name = source.Name  
    # WHEN NOT MATCHED THEN  
    #     INSERT (UnitMeasureCode, Name)  
    #     VALUES (source.UnitMeasureCode, source.Name)  

    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully. Also these Variables were loaded: {sql_name}, {sql_pw}.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
