import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sec_api import QueryApi

with open('C:\dev\pynance\secrets.json','r') as f:
    data = json.load(f)
    sec_api_key = data['sec_api_key']
    sec_io_key = data['sec_api_io_key']

    f.close()



class Filings():

    # get XBRL-JSON for a given accession number
    def get_xbrl_json( retry = 0):

        # get your API key at https://sec-api.io
        query_api = QueryApi(api_key=sec_io_key)

        # fetch all 10-Q and 10-K filings for Apple
        query = {
            "query": {
                "query_string": {
                    "query": "(formType:\"10-Q\" OR formType:\"10-K\") AND ticker:AAPL"
                }
            },
            "from": "0",
            "size": "20",
            "sort": [{ "filedAt": { "order": "desc" } }]
        }

        query_result = query_api.get_filings(query)
        print(query_result)
        accession_numbers = []

        # extract accession numbers of each filing
        for filing in query_result['filings']:
            accession_numbers.append(filing['accessionNo']);
        
        print(accession_numbers)

    #     request_url = xbrl_converter_api_endpoint + "?accession-no=" + accession_no + "&token=" + api_key

    #     # linear backoff in case API fails with "too many requests" error
    #     try:
    #     response_tmp = requests.get(request_url)
    #     xbrl_json = json.loads(response_tmp.text)
    #     except:
    #     if retry > 5:
    #         raise Exception('API error')
        
    #     # wait 500 milliseconds on error and retry
    #     time.sleep(0.5) 
    #     return get_xbrl_json(accession_no, retry + 1)

    #     return xbrl_json

Filings().get_xbrl_json()


class Extract():

    def __init__(self):
        filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019321000056/aapl-20210327.htm"
        xbrl_converter_api_endpoint = "https://api.sec-api.io/xbrl-to-json"
        api_key = sec_api_key # replace with your API key
        final_url = xbrl_converter_api_endpoint + "?htm-url=" + filing_url + "&token=" + api_key
        response = requests.get(final_url)
        self.xbrl_json = json.loads(response.text)
        # print(self.xbrl_json.keys()) # list all keys

        # print(self.xbrl_json['StatementsOfIncome'])
        # income_statement = self.get_income_statement(self.xbrl_json)
        # print(income_statement)
        # balance_sheet = self.get_balance_sheet(self.xbrl_json)
        # print(balance_sheet)
        # cash_flows = self.get_cash_flow_statement(self.xbrl_json)
        # print(cash_flows)
        # commentary = self.get_commentary_text()
        # print(commentary)

        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(self.xbrl_json, f, ensure_ascii=False, indent=4)



    def get_income_statement(self, xbrl_json):
        income_statement_store = {}

        # iterate over each US GAAP item in the income statement
        for usGaapItem in self.xbrl_json['StatementsOfIncome']:
            values = []
            indicies = []

            for fact in self.xbrl_json['StatementsOfIncome'][usGaapItem]:
                # only consider items without segment. not required for our analysis.
                if 'segment' not in fact:
                    index = fact['period']['startDate'] + '-' + fact['period']['endDate']
                    # ensure no index duplicates are created
                    if index not in indicies:
                        values.append(fact['value'])
                        indicies.append(index)                    

            income_statement_store[usGaapItem] = pd.Series(values, index=indicies) 

        income_statement = pd.DataFrame(income_statement_store)
        # switch columns and rows so that US GAAP items are rows and each column header represents a date range
        return income_statement.T 


    # convert XBRL-JSON of balance sheet to pandas dataframe
    def get_balance_sheet(self, xbrl_json):
        balance_sheet_store = {}

        for usGaapItem in self.xbrl_json['BalanceSheets']:
            values = []
            indicies = []

            for fact in self.xbrl_json['BalanceSheets'][usGaapItem]:
                # only consider items without segment.
                if 'segment' not in fact:
                    index = fact['period']['instant']

                    # avoid duplicate indicies with same values
                    if index in indicies:
                        continue
                        
                    # add 0 if value is nil
                    if "value" not in fact:
                        values.append(0)
                    else:
                        values.append(fact['value'])

                    indicies.append(index)                    

                balance_sheet_store[usGaapItem] = pd.Series(values, index=indicies) 

        balance_sheet = pd.DataFrame(balance_sheet_store)
        # switch columns and rows so that US GAAP items are rows and each column header represents a date instant
        return balance_sheet.T


    def get_cash_flow_statement(self, xbrl_json):
        cash_flows_store = {}

        for usGaapItem in self.xbrl_json['StatementsOfCashFlows']:
            values = []
            indicies = []

            for fact in self.xbrl_json['StatementsOfCashFlows'][usGaapItem]:        
                # only consider items without segment.
                if 'segment' not in fact:
                    # check if date instant or date range is present
                    if "instant" in fact['period']:
                        index = fact['period']['instant']
                    else:
                        index = fact['period']['startDate'] + '-' + fact['period']['endDate']

                    # avoid duplicate indicies with same values
                    if index in indicies:
                        continue

                    if "value" not in fact:
                        values.append(0)
                    else:
                        values.append(fact['value'])

                    indicies.append(index)                    

            cash_flows_store[usGaapItem] = pd.Series(values, index=indicies) 


        cash_flows = pd.DataFrame(cash_flows_store)
        return cash_flows.T


    def get_commentary_text(self):
        commentary_dict = {}
        for xkey in self.xbrl_json.keys():
            result = self.xbrl_json[xkey]
            for k,v in result.items():
                if isinstance(v, str):
                    soup = BeautifulSoup(v)
                    commentary_dict[k] = soup.get_text()
        return commentary_dict

    import time

