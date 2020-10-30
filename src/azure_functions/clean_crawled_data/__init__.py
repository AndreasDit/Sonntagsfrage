import logging
import pyodbc
import datetime as dt
import os
import azure.functions as func

def execute_sql_stmt(sql_stmt, cursor, conn):
    cursor.execute(sql_stmt)
    conn.commit()

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function cleans crawled data in the SQL DB.')
    # setup()

    # import variables
    name = req.params.get('name')
    sql_name = os.getenv('sql_db_name')
    sql_pw = os.getenv('sql_db_pw')
    # sql_name = os.environ['sql_db_name']
    # sql_pw = os.environ['sql_db_pw']

    # set defaults for azure sql datbse
    server = 'sonntagsfrage-server.database.windows.net'
    database = 'sonntagsfrage-sql-db'
    username = sql_name
    password = sql_pw 
    driver = '{ODBC Driver 17 for SQL Server}'

    # open connection
    conn_str = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password
    logging.info(conn_str)
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # empty target table
    sqlstmt = """truncate table sonntagsfrage.results_questionaire_clean"""
    execute_sql_stmt(sqlstmt, cursor, conn)
    logging.info('Target table sonntagsfrage.results_questionaire_clean successfully emptied.')

    # insert transformed values into clean table
    sqlstmt = """insert into sonntagsfrage.results_questionaire_clean
        select  
        Datum
        ,cast( coalesce( '0', replace(isnull(CDU_CSU  , '0'), '-', '0')) as numeric) CDU_CSU
        ,cast( coalesce( '0', replace(isnull(SPD      , '0'), '-', '0')) as numeric) SPD          
        ,cast( coalesce( '0', replace(isnull(GRUENE   , '0'), '-', '0')) as numeric) GRUENE      
        ,cast( coalesce( '0', replace(isnull(FDP      , '0'), '-', '0')) as numeric) FDP        
        ,cast( coalesce( '0', replace(isnull(LINKE    , '0'), '-', '0')) as numeric) LINKE        
        ,cast( coalesce( '0', replace(isnull(PIRATEN  , '0'), '-', '0')) as numeric) PIRATEN      
        ,cast( coalesce( '0', replace(isnull(AfD      , '0'), '-', '0')) as numeric) AfD       
        ,cast( coalesce( '0', replace(isnull(Linke_PDS, '0'), '-', '0')) as numeric) Linke_PDS    
        ,cast( coalesce( '0', replace(isnull(PDS      , '0'), '-', '0')) as numeric) PDS          
        ,cast( coalesce( '0', replace(isnull(REP_DVU  , '0'), '-', '0')) as numeric) REP_DVU  
        , cast( coalesce( '0', replace(replace(replace( isnull(Sonstige , '0'), '-', '0'), 'PIR', '0'), 'WASG3', '0')) as numeric) Sonstige
        ,Befragte   
        ,Zeitraum
        ,''
        from sonntagsfrage.results_questionaire q
        """
    execute_sql_stmt(sqlstmt, cursor, conn)
    logging.info('Cleaned values successfully inserted into table sonntagsfrage.results_questionaire_clean.')


    return func.HttpResponse(
             "This HTTP triggered function did transform the data in the Azuer SQL DB.",
             status_code=200
        )