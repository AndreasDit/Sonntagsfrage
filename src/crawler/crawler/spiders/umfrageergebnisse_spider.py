import scrapy
import datetime

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
        TABLE_SELECTOR = 'td[4]//text()'
        page = response.css(TABLE_SELECTOR)
        # filename = 'ergebnisse-%s.html' % page
        filename = 'ergebnisse-%s.html' % str(datetime.now())
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

#         SET_SELECTOR = '.push-data'
#         TABLE_SELECTOR_old = '//div[@class="table-responsive relative"]'
#         ROWS_SELECTOR_old = '//div[@class="table-responsive relative"]//tr'
#         ROW_SELECTOR = '//tr'
#
#         TABLE_SELECTOR = 'table class="wilko"'
#
#         table = response.xpath(TABLE_SELECTOR)
#         rows = response.xpath(ROWS_SELECTOR)
#         rows = rows[1: len(rows) - 3]
#
#         for row_set in rows:
# #        for row_set in response.css(SET_SELECTOR):
#
#             VORTAG_SELECTOR     = 'td[4]//text()'
#             BID_SELECTOR        = 'td[5]//text()'
#             ASK_SELECTOR        = 'td[6]//text()'
#             TIMESTAMP_SELECTOR  = 'td[9]//text()'
#             NAME_SELECTOR       = '::text'
#             yield {
#                 'name': row_set.css(NAME_SELECTOR).extract_first(),
#                 'vortag': row_set.xpath(VORTAG_SELECTOR).extract_first(),
#                 'bid': row_set.xpath(BID_SELECTOR).extract_first(),
#                 'ask': row_set.xpath(ASK_SELECTOR).extract_first(),
#                 'timestamp': row_set.xpath(TIMESTAMP_SELECTOR).extract_first(),
#             }


