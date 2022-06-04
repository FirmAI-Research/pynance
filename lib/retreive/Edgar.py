'''
use sec retrievers to parse sec edgar filings and populate sql database for nlp
'''

class Edgar():

    def parse_rss(self):
        pass

    def parse_xbrl(self):
        pass



from sec_edgar_downloader import Downloader
import os

def download_latest_filing(file_type, ticker):
	dl = Downloader(os.getcwd())
	dl.get(file_type, ticker, amount=1)
	dl_path = os.getcwd() +'/sec-edgar-filings/{}/{}/'.format(ticker, file_type)

	inner_most_dir = [x[0] for x in os.walk(dl_path)][1]
	html_path = f'{inner_most_dir}/filing-details.html'
	txt_path = f'{inner_most_dir}/full-submission.txt'

	return (html_path, txt_path)

# html, txt = download_latest_filing("8-K","RPLA")
# print(html)
# print(txt)

from sec_edgar_retrievers import edgar_downloader as edl
from bs4 import BeautifulSoup
import sys,os

file_type = "10-K"
ticker = "RPLA"
html, txt = edl.download_latest_filing(file_type,ticker)

myfile = open(html)
contents = myfile.read()
myfile.close()

soup = BeautifulSoup(contents, features='lxml')
VALID_TAGS = ['div', 'p']

for tag in soup.findAll('p'):
        if tag.name not in VALID_TAGS:
            tag.replaceWith(tag.renderContents())
        for attribute in ["class", "id", "name", "style", "td","tr"]:
            del tag[attribute]
        if tag.parent.name == 'p' and tag.name not in ["script", "table"]:
            tag.getText()
soup_out = soup.renderContents()

dir = os.getcwd() + f'/output/{ticker}/'
if not os.path.isdir(dir):
    os.mkdir(dir)
outp = dir + 'soup.txt'
f = open(outp, 'wb')
f.write(soup_out)
f.close()
