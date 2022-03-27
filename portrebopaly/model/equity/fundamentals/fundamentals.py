from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  Tickers


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