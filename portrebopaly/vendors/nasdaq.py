""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                                            Nasdaq Data Link                                  
  
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

from curses import keyname
import nasdaqdatalink
import quandl
import json
import os

class Nasdaq:
    """
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31')
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31', ticker='ZYXI')
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.csv?api_key=API_KEY?download_type=complete

    """    

    # equtiy
    core_us_fundamentals = 'SHARADAR/SF1'


    def __init__(self):
        self.authenticate()


    def authenticate(self):        
        with open('../secrets.json') as f:
            data = json.load(f)
        nasdaqdatalink.read_key()
        os.environ["NASDAQ_DATA_LINK_API_KEY"] =  data['nasdaq_api_key']  # NOTE options: nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key']   |     


    def get(self, name, ticker=None, date = None):
        # if ticker is None:
        #     return nasdaqdatalink.get_table(name, paginate=True)
        # else:
        #     return nasdaqdatalink.get_table(name, ticker=ticker)
        df = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31', ticker = 'AMZN', paginate=True)
        df.to_csv('.\\vendors\\exports')