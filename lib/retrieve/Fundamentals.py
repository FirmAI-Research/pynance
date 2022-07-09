import numpy as np 
import pandas as pd 

import pandas as pd
import numpy as np 
import datetime
import os
from pandas_datareader import data as web
from yahoo_fin.stock_info import *
import yfinance as yfin
from utils import yahoo_api_utils as yahooutils
import re
import json
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime
now = datetime.now()
import sys, os  
import matplotlib.pyplot as plt 
from sqlalchemy import create_engine
import requests
import csv
cwd = os.getcwd()
from pathlib import Path


cwd = os.getcwd()
from pathlib import Path

p = f'{Path(os.getcwd()).parent.absolute()}/'
statements_abrev_names = ['qbs','qis','qcf', 'abs','ais','acf']


def download_financial_statements_to_local(ticker, extracted_dict):
    """
    function to download financial statements for input ticker

    """
    for k,v in extracted_dict.items(): # for each k = sheet type, v = []
        localdir = f'{p}/output/yahoo_fundamentals/{ticker}/'  
        if not os.path.isdir(localdir):
            os.mkdir(localdir)
        
        localp = f'{localdir}{ticker}_{k}.csv'
        with open(localp, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for _k,_v in v[0].items():
                writer.writerow([_k, _v])
    print('download successfull for ' + str(extracted_dict.keys()))



# !
def get_yahoo_financial_statements(tickers = [], download = False, upload = False):
    """
    ________________
    Parse yahoo api via json urls to retrieve financial statement data

    ________________
    :param w: 
    :type w: 
    :param obj: 
    :type obj: 
    :return: 
    :rtype: 
    ________________
    """
    if tickers != None:
        for ticker in tickers:
            response = yahooutils.build_financial_statements_urls(ticker)
            soup = BeautifulSoup(response.text, 'html.parser')
            data_dict = yahooutils.parse_financial_statements_json_response(soup)
            extracted_dict = dict() # parse json response from yahoo api
            for k,v in data_dict.items(): #keys = xdict returned from util.parse_json()
                parsed_statements = []
                for s in v:
                    statement = {}
                    for _k,_v in s.items(): # line items from statements
                        try:
                            statement[_k] = _v['raw']
                        except TypeError:
                            continue
                        except KeyError:
                            continue
                        parsed_statements.append(statement)
                extracted_dict[k] = parsed_statements
        
            if download == True:
                download_financial_statements_to_local(ticker, extracted_dict)
            elif upload == True:
                upload_financial_statements_to_db(ticker)
    else:
        print('Unable to parse yahoo')
    



# use yahoo_fin

def get_balance_sheet():
    pass
def get_cash_flow():
    get_cash_flow('nflx', yearly = False)
    pass
