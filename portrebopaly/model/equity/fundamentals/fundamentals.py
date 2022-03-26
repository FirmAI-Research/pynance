from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  Tickers

def filter_by_sector(sector:str):
    """ filter a dataframe of equity fundamentals for all companies with in a sector
    
    """
    core = CoreUsFundamentals()
    df_core =  core.get_export()
    df = core.merge_meta_data(df_core)
    return  df.loc[df.sector==sector]