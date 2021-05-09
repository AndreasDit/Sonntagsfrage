import logging
import pyodbc
import datetime as dt
import os
import scrapy
import json
# import from crochet import setup
from twisted.internet import reactor
from scrapy.crawler import CrawlerProcess, CrawlerRunner 
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
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
    logging.info(sql_pw)
    logging.info(sql_name)

    # open connection
    conn_str = 'DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password
    logging.info(conn_str)
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()

    # define spider
    class UmfrageerbegnisseSpider(scrapy.Spider):
        name = "umfrageergebnisse"

        def start_requests(self):
            urls = [
                'https://www.wahlrecht.de/umfragen/dimap/1998.htm',
                'https://www.wahlrecht.de/umfragen/dimap/1999.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2000.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2001.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2002.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2003.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2004.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2005.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2006.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2007.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2008.htm',
                'https://www.wahlrecht.de/umfragen/dimap/2013.htm',
                'https://www.wahlrecht.de/umfragen/dimap.htm'
            ]
            for url in urls:
                yield scrapy.Request(url=url, callback=self.parse)

        def parse(self, response):
            # page = response.url.split("/")[-2]
            s_timestamp = str(dt.datetime.now()).replace(':', '-')
            filename = 'umfrageergebnisse-%s.csv' % s_timestamp

            # set defaults
            sep = ','

            # extract relevant data
            # parties_description = response.xpath('//td[@id="abkuerzungen-parteien"]//span/text()').getall()
            # parties_voted_old = response.xpath('//thead//a/text()').getall()
            parties_voted = response.xpath('//thead//th[@class="part"]//text()').getall()
            colname_other_parties = response.xpath('//thead//a[@href="#fn-son"]//text()').getall()
            if len(colname_other_parties) == 0: colname_other_parties = ['Sonstige']
            colname_people_asked = response.xpath('//thead//th[@class="befr"]/text()').getall()
            if len(colname_people_asked) == 0: colname_people_asked = ['Befragte']
            colname_time_from_to = response.xpath('//thead//th[@class="dat2"]/text()').getall()
            if len(colname_time_from_to) == 0: colname_time_from_to = ['Zeitraum']

            # clean up for CDU/CSU
            parties_voted_fixed = self.clean_for_double_party_names(parties_voted)
            parties_voted_fixed = [party.replace('/', '_').replace('.', '_') for party in parties_voted_fixed]
            if parties_voted_fixed[-1] == 'Sonstige':
                parties_voted_fixed = parties_voted_fixed[:-1]
            nb_parties = len(parties_voted_fixed) + 1

            # create haeder for csv-file with all col_names
            colname_date_published = ['Datum']
            header_array = colname_date_published + parties_voted_fixed + colname_other_parties + colname_people_asked + colname_time_from_to
            header_string = sep.join(header_array)
            header_string = self.clean_umlaute(header_string)

            # crawl the actual data
            table_selctor = '//tbody//tr'
            table = response.xpath(table_selctor)

            for rows in table:
                # start the data-string with the publishing date
                date_selector = 'td[1]//text()'
                date = rows.xpath(date_selector).get()
                data_string = "'" + date + "'"

                # add the questionaire results for each party
                for party_idx in range(0, nb_parties):
                    actual_party_idx_in_url = party_idx + 3
                    party_selector = 'td[' + str(actual_party_idx_in_url) + ']//text()'
                    party_percentage = rows.xpath(party_selector).get()
                    party_percentage_cleaned = party_percentage.replace(' ', '').replace('%', '').replace(',', '.')

                    data_string += sep + "'" + party_percentage_cleaned + "'"
                    data_string = self.clean_umlaute(data_string)

                # add meta info about the number of participiants
                people_asked_selector = 'td[@class="s"][2]//text()'
                people_asked = rows.xpath(people_asked_selector).get()
                if people_asked is not None:
                    people_asked_cleaned = people_asked.replace('.', '').replace(',', '').replace(' ', '')
                else:
                    people_asked_cleaned = ''
                data_string += sep + "'" + people_asked_cleaned + "'"

                # add meta info about timebox in which the questioning took place
                time_from_to_selector = 'td[@class="s"][3]//text()'
                time_from_to = rows.xpath(time_from_to_selector).get()
                if time_from_to is None:
                    time_from_to = ''
                data_string += sep + "'" + time_from_to + "'"

                # delete existing row
                sqlstmt = """delete from  sonntagsfrage.results_questionaire
                    where Datum = '""" + date + """'"""
                cursor.execute(sqlstmt)
                conn.commit()

                # send datarow to azure sql db
                sqlstmt = """insert into sonntagsfrage.results_questionaire(
                    """ + header_string + """) values (
                    """ + data_string + """
                    )"""
                # logging.info(header_string)
                # logging.info(sqlstmt)
                cursor.execute(sqlstmt)
                conn.commit()

            self.log('Saved file %s' % filename)

        def clean_for_double_party_names(self, parties_voted):
            """ Scraping the site yields: ['CDU', '/', 'CSU'].
                This Methods combines these to: ['CDU/CSU'] and combines with other party names to an array.
            """
            try:
                cdu_cdu_pos = parties_voted.index('/')
                if cdu_cdu_pos is None:
                    cdu_cdu_pos = parties_voted.index('.')
            except:
                return parties_voted

            cdu_csu = parties_voted[cdu_cdu_pos - 1:cdu_cdu_pos + 2]
            cdu_csu_string = ''.join(cdu_csu)
            parties_voted_fixed = parties_voted[0:cdu_cdu_pos - 1] \
                                + [cdu_csu_string] \
                                + parties_voted[cdu_cdu_pos + 2:len(parties_voted)]

            return parties_voted_fixed

        def clean_umlaute(self, input):
            replacers = {'ä': 'ae', 'ö': 'oe',
                        'ü': 'ue', 'ß': 'ss',
                        'Ä': 'AE', 'Ö': 'OE',
                        'Ü': 'UE', '–': '-'
                        }
            for key, value in replacers.items():
                input = input.replace(key, value)

            return input


    process = None
    process = CrawlerRunner()
    crawler = process.crawl(UmfrageerbegnisseSpider)

    crawler.addBoth(lambda _: reactor.stop())
    reactor.run() # the script will block here until the crawling is finished


    return func.HttpResponse(
            json.dumps({
            'response':"This HTTP triggered function did crawl the required sites and wrote the resultst into an Azuer SQL DB."
            }),
            status_code=200
        )

    

    ###### useful sql snippets #####

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