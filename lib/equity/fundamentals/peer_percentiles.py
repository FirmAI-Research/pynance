import pandas as pd
import numpy as np
import nasdaqdatalink
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
from lib.nasdaq import Fundamentals, Metrics, Tickers, Nasdaq
from lib.calendar import Calendar
cal = Calendar()
from dateutil.relativedelta import relativedelta


import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

'''

populates tables for each sector and industry of peer percentiles and ranks for the MOST RECENT QUARTER
'''

def get_numeric_cols(df):
    return df.select_dtypes(exclude=['object', 'datetime64']).columns


def calculate_box_plot(df, column=None):
    ''' calculat quartiles from an array '''
    arr = pd.to_numeric(df[column]).dropna(axis=0, how='any').values
    try:
        Q1, median, Q3 = np.percentile(arr, [25, 50, 75])
        IQR = Q3 - Q1
        loval = Q1 - 1.5 * IQR
        hival = Q3 + 1.5 * IQR
        wiskhi = np.compress(arr <= hival, arr)
        wisklo = np.compress(arr >= loval, arr)
        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)
        outliers_high = np.compress(arr > actual_hival, arr)
        outliers_low = np.compress(arr < actual_loval, arr)
        outliers = [x for x in outliers_high ] + [y for y in outliers_low]
        Qs = [actual_loval, Q1, median, Q3, actual_hival]
        return Qs, outliers
    except IndexError as e:
        return [], []


def build_percentiles_frame(df):
    numeric_cols = get_numeric_cols(df)
    datalists = []
    for c in df.columns:
        if c in numeric_cols:
            qs, outliers = calculate_box_plot(df, column=c)
            datalists.append(qs)
    xdf = pd.DataFrame(datalists, columns=['low', 'Q1', 'median', 'Q3', 'high']).transpose() # use pd.concat
    xdf.columns = [x for x in numeric_cols]
    for c in xdf.columns:
        xdf[c] = xdf[c].apply(lambda x : '{:,.2f}'.format(x))
    xdf.reset_index(inplace=True)
    xdf.replace('nan',np.nan, inplace=True)
    xdf.dropna(axis=1, how='any', inplace=True)
    return xdf


def build_ranks_by_company_frame(fundamentals):
    frames = []
    for c in get_numeric_cols(fundamentals):
        frames.append(fundamentals[c].rank(pct=True, ascending = True))
    ranks_by_company = pd.concat(frames, axis=1)
    ranks_by_company['ticker'] = fundamentals['ticker']#.iloc[:, :1]
    ranks_by_company = ranks_by_company[ranks_by_company.columns.tolist()[-1:] + ranks_by_company.columns.tolist()[:-1]]
    for c in get_numeric_cols(ranks_by_company):
        ranks_by_company[c] = ranks_by_company[c].apply(lambda x : '{:,.2f}'.format(float(x)))
    return ranks_by_company


def fundamental_peer_percentiles():
    ''' calculates the percentile rank for every company in a sector for each metric reported in their financial statemets
    '''
    tickers = Tickers().get() 
    tickers = tickers.loc[ ( pd.to_datetime(tickers.lastpricedate) >= pd.to_datetime(cal.previous_quarter_end()) ) & (tickers.currency == 'USD')]
    tickers = tickers[['ticker','name','cusips','sector','industry', 'sicsector', 'sicindustry',  'famaindustry']]

    fun = Fundamentals(calendardate=cal.previous_quarter_end())
    fundamentals = fun.get()
    df = tickers.merge(fundamentals, how='left', on='ticker')

    unique_sectors_list = df.sector.unique().tolist()
    unique_industries_list = df.industry.unique().tolist()

    # @Sectors
    sector_percentile_frames = {}
    sector_rank_frames ={}

    for sector_str in unique_sectors_list:
        try:
            asector = df.loc[df.sector == sector_str]
            sector_prcentiles = build_percentiles_frame(asector)
            sector_prcentiles['sector'] = sector_str
            sector_prcentiles['date'] = cal.previous_quarter_end()
            sector_prcentiles['uid'] = sector_str +  ' ' + str(cal.previous_quarter_end())

            company_ranks = build_ranks_by_company_frame(asector)
            company_ranks['sector'] = sector_str
            company_ranks['date'] = cal.previous_quarter_end()
            company_ranks['uid'] = sector_str +  ' ' + str(cal.previous_quarter_end())

            # engine = Postgres().engine
            # sector_prcentiles.to_sql(f'Percentiles_Sector', engine, if_exists='append')
            # company_ranks.to_sql(f'Ranks_Sector', engine,  if_exists='append')

            sector_percentile_frames[sector_str] = sector_prcentiles
            sector_rank_frames[sector_str] = company_ranks

            # print('[SUCCESS] Sector Percentiles and Ranks for {}'.format(sector_str))

        except Exception as e:
            # print(f'*****[ERROR]***** {e}')            
            pass       

    # @Industry

    industry_percentile_frames = {}
    industry_rank_frames ={}

    for industry_str in unique_industries_list:
        try:
                
            aindustry= df.loc[df.industry == industry_str]
            industry_prcentiles = build_percentiles_frame(asector)
            industry_prcentiles['industry'] = industry_str
            industry_prcentiles['date'] = cal.previous_quarter_end()
            industry_prcentiles['uid'] = industry_str +  ' ' + str(cal.previous_quarter_end())

            company_ranks = build_ranks_by_company_frame(aindustry)
            company_ranks['industry'] = industry_str
            company_ranks['date'] = cal.previous_quarter_end()
            company_ranks['uid'] = industry_str +  ' ' + str(cal.previous_quarter_end())

            industry_percentile_frames[industry_str] = industry_prcentiles
            industry_rank_frames[industry_str] = company_ranks

        except Exception as e:
            # print(f'*****[ERROR]***** {e}')   
            pass       
    # 

    return unique_sectors_list, unique_industries_list, sector_percentile_frames, sector_rank_frames , industry_percentile_frames, industry_rank_frames


