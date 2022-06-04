import requests
import re 
import json


def build_financial_statements_urls(ticker):
	url_stats = 'https://finance.yahoo.com/quote/{}/key-statistics?p={}'
	url_profile = 'https://finance.yahoo.com/quote/{}/profile?p={}'
	url_financials = 'https://finance.yahoo.com/quote/{}/financials?p={}'

	response = requests.get(url_financials.format(ticker, ticker))
	return response


def parse_financial_statements_json_response(soup):
	pattern = re.compile('\s--\sData\s--\s')
	data = soup.find('script', text = pattern).contents[0]
	start = data.find('context')-2
	json_data = json.loads(data[start:-12])
	json_data['context'].keys()
	json_data['context']['dispatcher']['stores']['QuoteSummaryStore'].keys()
	annual_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['incomeStatementHistory']['incomeStatementHistory']
	quarterly_is = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['incomeStatementHistoryQuarterly']['incomeStatementHistory']

	annual_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['cashflowStatementHistory']['cashflowStatements']
	quarterly_cf = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['cashflowStatementHistoryQuarterly']['cashflowStatements']

	annual_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['balanceSheetHistory']['balanceSheetStatements']
	quarterly_bs = json_data['context']['dispatcher']['stores']['QuoteSummaryStore']['balanceSheetHistoryQuarterly']['balanceSheetStatements']
	
	# store k,v pairs as dict to pass response and extract values
	xdict = {'ais':annual_is, 'qis':quarterly_is, 'acf':annual_cf, 'qcf':quarterly_cf, 'abs':annual_bs, 'qbs':quarterly_bs}
	return xdict


