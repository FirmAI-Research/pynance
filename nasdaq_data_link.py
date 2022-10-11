""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                      Nasdaq Data Link                                              │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
"""

import re
import nasdaqdatalink
import json
import sys, os
import pandas as pd
import numpy as np
import requests

from enum import Enum

proj_root = os.path.dirname(os.path.abspath(__file__))

from calendar_dates import Calendar
cal = Calendar()


class Sharadar(Enum):
  
  METRICS = 'SHARADAR/Daily'
  FUNDAMENTALS = 'SHARADAR/SF1'
  TICKERS = 'SHARADAR/TICKERS'
  INSTITUIONS = 'SHARADAR/SF3'
  INSIDERS =  'SHARADAR/SF2'


class Nasdaq:
  ''' Retreives data from nasdaq-data-link api

  API Call
    nasdaqdatalink.get_table(table, ticker = ticker, date = {'gte': date_start.strftime('%Y-%m-%d'), 'lte': date_end.strftime('%Y-%m-%d') }, paginate=True) 
    nasdaqdatalink.get_table(table, ticker = ticker, date=date, paginate=True) 
    nasdaqdatalink.get_table(table, date=[cal.previous_quarter_end()], paginate=True)

  Export URLS
    To bypass the 10,000 row export limit: qopts.export=True; Returns an excel file with the link to an S3 bucket in cell A1; Curl the link to download a zip of the data.
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF3.csv?&calendardate=2021-09-30&api_key=API_KEY&qopts.export=true
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key=API_KEY

  '''
  def __init__(self):
    
    self.authenticate()


  def authenticate(self):    

    fp = os.path.join(proj_root, 'nasdaq_data_link.json')
    
    with open(fp) as f:
        data = json.load(f)
    
    self.api_key = data['nasdaq_api_key'] 
    
    nasdaqdatalink.ApiConfig.api_key = self.api_key




