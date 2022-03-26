
import pandas as pd
import nasdaqdatalink
from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  Tickers

# -- NasdaqDataLink Api Call --
def test_api_call():
    core = CoreUsFundamentals()
    df = core.get()
    print(df)


# -- CoreUsFundamentals sample export -- 
def test_core_us_fundamentals():
    core = CoreUsFundamentals()
    df_core =  core.get_export()
    df = core.merge_meta_data(df_core)
    df = df.loc[df.sector=='Technology']
    print(df)
    
    return df


test_core_us_fundamentals()


# -- CoreUsFundamentals sample export -- 
def test_tickers():
    tickers = Tickers()
    df =  tickers.get_export()
    print(df)
# test_tickers()