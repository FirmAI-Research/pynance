import numpy as np
import pandas as pd

from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  Tickers

class Semiconductors(Nasdaq):

    def __init__(self):
        super().__init__()

        self.colnames = static_cols + ['revenue', 'ebitda', 'ebit', 'capex', 'debt', 'fcf', 'intexp']  # Core US Fundamentals columns:    debt = total debt; intexp = interest expense
        self.calc_colnames = ['ebitda_margin', 'ebitda_less_capex_margin', 'debt_to_ebitda', 'fcf_to_debt', 'ebit_to_interest_expense']


    def calculate(self):
        ''' calculated fundamental metrics/ratios
        '''
        self.df['ebitda_margin'] = self.df['ebitda'] / self.df['revenue']
        self.df['ebitda_less_capex_margin'] = (self.df['ebitda'] + self.df['capex']) / self.df['revenue']  
        self.df['debt_to_ebitda'] = self.df['debt'] / self.df['ebitda'] 
        self.df['fcf_to_debt'] = self.df['fcf'] / self.df['debt'] 
        self.df['ebit_to_interest_expense'] = self.df['ebit'] / self.df['intexp'] 


    def build_table(self, sector:str=None, industry:str=None):
        ''' filters US Core Fundamentals data by sector and industry. applies number formatting to report in millions. performs calculations on the raw fundamentals
        returns an dataframe with a concatenation of the static, raw, and calculated columns.
        '''
        self.df = filter_by_sector_and_industry(sector, industry)[self.colnames]
        self.df = number_format(self.df)
        self.calculate()
        self.df = self.df[static_cols + self.calc_colnames + [c for c in self.colnames if c not in static_cols] ].reset_index(drop=True)
        return self.df



########### utility & helper functions

static_cols = ['ticker', 'name', 'calendardate', 'reportperiod','sector','industry']

def number_format(df):
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number): # check if data is a sub data type of numpy.number
            df[c] = df[c].apply(lambda x: x/1000000)  # divide by 1 million to report value in millions; '{:,}'.format() prevents sorting in the gui
    return df


def list_sectors():
    """ list Core US Equity fundamentals sectors
    """
    tickers = Tickers()
    df =  tickers.get_export()
    return df.sector.unique().tolist()


def list_industries(sector:str):
    """ list all industries for a given sector
    """
    tickers = Tickers()
    df =  tickers.get_export()
    df = df.loc[df.sector == sector]
    return df.industry.unique().tolist()


def filter_by_sector(sector:str):
    """ filter Core US Equity fundamentals by sector
    """
    core = CoreUsFundamentals()
    df_core =  core.get_export()
    df = core.merge_meta_data(df_core)
    return  df.loc[df.sector==sector]


def filter_by_sector_and_industry(sector:str, industry:str):
    """ filter Core US Equity fundamentals by sector and industry
    """
    core = CoreUsFundamentals()
    df_core =  core.get_export()
    df = core.merge_meta_data(df_core)
    return  df.loc[(df.sector==sector) & (df.industry==industry)]



    