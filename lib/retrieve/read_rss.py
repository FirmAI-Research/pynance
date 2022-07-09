'''
@ michael 03-2021
parse rss feeds & return a dataframe of edgar sec filing data

https://www.sec.gov/Archives/edgar/xbrlrss.all.xml
https://www.sec.gov/Archives/edgar/xbrl-inline.rss.xml
https://www.sec.gov/Archives/edgar/usgaap.rss.xml
https://www.sec.gov/Archives/edgar/xbrl-inline.rss.xml
'''
from bs4 import BeautifulSoup
import feedparser
import pandas as pd 

xdict = dict()
outdict = dict()
def getTagInfo(key, url, count=0):

    def parseRSS(url):
        return feedparser.parse(url)
    feed = parseRSS(url)

    for i in feed['items']:
        xdict[count] = [i['title'], i['link']]
        count+=1

    df = pd.DataFrame.from_dict(xdict).transpose()
    outdict[key] =df



newsurls = {
        'PressRelease': 'https://www.federalreserve.gov/feeds/press_all.xml',
        'Speeches':'https://www.federalreserve.gov/feeds/speeches.xml',
        'Testimony':'https://www.federalreserve.gov/feeds/testimony.xml',
        'Data':'https://www.federalreserve.gov/feeds/datadownload.xml',
}
for k,v in newsurls.items():
    getTagInfo(k,v)

# print(outdict.get('PressRelease'))

