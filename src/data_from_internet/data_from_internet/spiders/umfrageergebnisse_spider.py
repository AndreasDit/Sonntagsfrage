import scrapy
import datetime as dt
import csv


class QuotesSpider(scrapy.Spider):
    name = "umfrageergebnisse"

    def start_requests(self):
        urls = [
            'https://www.wahlrecht.de/umfragen/dimap.htm'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # page = response.url.split("/")[-2]
        s_timestamp = str(dt.datetime.now()).replace(':', '-')
        filename = 'umfrageergebnisse-%s.csv' % s_timestamp

        # set defaults
        sep = ';'

        # extract relevant data
        parties_description = response.xpath('//td[@id="abkuerzungen-parteien"]//span/text()').getall()
        parties_voted_old = response.xpath('//thead//a/text()').getall()
        parties_voted = response.xpath('//thead//th[@class="part"]//text()').getall()
        colname_other_parties = response.xpath('//thead//a[@href="#fn-son"]//text()').getall()
        colname_people_asked = response.xpath('//thead//th[@class="befr"]/text()').getall()
        colname_time_from_to = response.xpath('//thead//th[@class="dat2"]/text()').getall()

        # clean up for CDU/CSU
        parties_voted_fixed = self.clean_for_double_party_names(parties_voted)
        nb_parties = len(parties_voted_fixed) + 1

        # create haeder for csv-file with all col_names
        colname_date_published = ['Datum']
        header_array = colname_date_published + parties_voted_fixed + colname_other_parties + colname_people_asked + colname_time_from_to
        header_string = sep.join(header_array)
        header_string = self.clean_umlaute(header_string)

        # write to file
        with open(filename, 'w') as f:
            writer = csv.writer(f)

            # start writing file with header
            f.write(header_string + '\r')

            # crawl the actual data
            table_selctor = '//tbody//tr'
            table = response.xpath(table_selctor)

            for rows in table:
                # start the data-string with the publishing date
                date_selector = 'td[1]//text()'
                date = rows.xpath(date_selector).get()
                data_string = date

                # add the questionaire results for each party
                for party_idx in range(0, nb_parties):
                    actual_party_idx_in_url = party_idx + 3
                    party_selector = 'td[' + str(actual_party_idx_in_url) + ']//text()'
                    party_percentage = rows.xpath(party_selector).get()
                    party_percentage_cleaned = party_percentage.replace(' ', '').replace('%', '').replace(',', '.')

                    data_string += sep + party_percentage_cleaned
                    data_string = self.clean_umlaute(data_string)

                # add meta info about the number of participiants
                people_asked_selector = 'td[@class="s"][2]//text()'
                people_asked = rows.xpath(people_asked_selector).get()
                if people_asked is not None:
                    people_asked_cleaned = people_asked.replace('.', '').replace(',', '').replace(' ', '')
                else:
                    people_asked_cleaned = ''
                data_string += sep + people_asked_cleaned

                # add meta info about timebox in which the questioning took place
                time_from_to_selector = 'td[@class="s"][3]//text()'
                time_from_to = rows.xpath(time_from_to_selector).get()
                if time_from_to is None:
                    time_from_to = ''
                data_string += sep + time_from_to

                print(data_string)
                # write final data to file
                f.write(data_string + '\r')

        self.log('Saved file %s' % filename)

    def clean_for_double_party_names(self, parties_voted):
        """ Scraping the site yields: ['CDU', '/', 'CSU'].
            This Methods combines these to: ['CDU/CSU'] and combines with other party names to an array.
        """

        cdu_cdu_pos = parties_voted.index('/')
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
