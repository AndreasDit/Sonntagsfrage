# imports
import pyodbc
import datetime as dt
import os


# default params
server = 'sonntagsfrage-server.database.windows.net'
database = 'sonntagsfrage-sql-db'
username = 'sonntagspredictor'
password = 'das_ist_EiN_sICHe_res_PasSworD_!'
driver = '{ODBC Driver 17 for SQL Server}'


# open connection
conn_str = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password
print(conn_str)
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# empty target table
sqlstmt = """truncate table sonntagsfrage.results_questionaire_clean"""
cursor.execute(sqlstmt)
conn.commit()

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
cursor.execute(sqlstmt)
conn.commit()



