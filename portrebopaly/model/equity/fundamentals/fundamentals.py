import numpy as np
import pandas as pd

from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  Tickers


class Moodys(Nasdaq):

    moody_ratings = ['Aaa', 'Aa', 'A', 'Baa', 'Ba', 'B', 'Caa', 'Ca']

    def __init__(self):
        super().__init__()

    def map_score(self, df_scorecard:pd.DataFrame, df_fundamentals:pd.DataFrame) -> pd.DataFrame:
        """takes a scorecard dataframe and US Core fundamentals df to returns the Moodys rating for associated fundamentals values
        
        :param [df_scorecard]: a df with fundamental metrics labels as indexes, and moodys rating as columns. values represent breakpoints for each metric to be assigned a given rating
        :param [df_fundamentals]: a df with fundamental data with rows representing fundamental data of an individual company and columns containing fundamental metrics.
            all fundamental metrics listed in the df scorecard index must be contained as a subset with in the df fundamentals columns

        :raises [ErrorType]: 
        """



class Semiconductors(Moodys):

    def __init__(self):
        super().__init__() 

        self.colnames = ['revenue', 'ebitda', 'ebit', 'capex', 'debt', 'fcf', 'intexp']  # Core US Fundamentals columns:    debt = total debt; intexp = interest expense
        self.calc_colnames = ['ebitda_margin', 'ebitda_less_capex_margin', 'debt_to_ebitda', 'fcf_to_debt', 'ebit_to_interest_expense']
        self.moodys_weights = [0.2, 0.05, 0.05, 0.1, 0.1, 0.05]


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
        self.df = filter_by_sector_and_industry(sector, industry)[static_cols + self.colnames]
        self.df = number_format(self.df)
        self.calculate()
        self.df = self.df[static_cols + self.calc_colnames + self.colnames ].reset_index(drop=True)
        return self.df


    def build_scorecard(self):
        scorecard = pd.DataFrame(columns = self.moody_ratings, index = ['revenue'] + self.calc_colnames)
        scorecard.loc['revenue', :]                     = [50, 30, 15, 5, 2, 0.75, 0.25, '<']   # ($, B)
        scorecard.loc['ebitda_margin', :]               = [0.5, 0.35, 0.3, 0.25, 0.2, 0.15, 0.10, '<']   # %
        scorecard.loc['ebitda_less_capex_margin', :]    = [0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05, '<']   # %
        scorecard.loc['debt_to_ebitda', :]              = [0.5, 1, 1.5, 2.5, 3.5, 5, 7, '>']   # 0x
        scorecard.loc['fcf_to_debt', :]                 = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, '<']   # %
        scorecard.loc['ebit_to_interest_expense', :]    = [30, 20, 10, 5, 3, 1.5, 0.0, '<']   # ($, B)
        moodys_weights = [0.2, 0.05, 0.05, 0.1, 0.1, 0.05] # buisness profile & financial policy are excluded as currently non-quantifiable
        scorecard['weights'] = moodys_weights
        print(scorecard)
        return scorecard.reset_index(drop=False)


class Insurance(Moodys):
    pass
























########### utility & helper functions for retreiving fundamental data from vendors


static_cols = ['ticker', 'name', 'currency', 'calendardate', 'reportperiod','sector','industry']

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



    