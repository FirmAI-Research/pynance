
import pandas as pd
import nasdaqdatalink
from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  CoreUSInstitutionalInvestors,  Tickers

# -- NasdaqDataLink Api Call --
def test_api_call():
    core = CoreUsFundamentals()
    df = core.get()
    print(df)
# test_api_call()

# -- CoreUsFundamentals sample export -- 
def test_core_us_fundamentals():
    core = CoreUsFundamentals()
    df_core =  core.get_export()
    df = core.merge_meta_data(df_core)
    df = df.loc[df.sector=='Technology']
    print(df)
    return df
# test_core_us_fundamentals()


# -- CoreUsFundamentals sample export -- 
def test_tickers():
    tickers = Tickers()
    df =  tickers.get_export()
    print(df.columns)
# test_tickers()



# -- CoreUSInstitutionalInvestors sample export -- 
def test_institutions():
    core = CoreUSInstitutionalInvestors()
    core.get_export()
    # print(core.get())
    # print(df.calendardate.value_counts())
    # print(df.columns)
    # print(raw_df.tail())
    # print(raw_df.shape)
    # print(core.group_by_ticker(raw_df))
    # core.group_by_institution()


    core.qtr_over_qtr_change(qtr_start = '2021-12-31', qtr_end = '2020-09-30')


test_institutions()





def test_build_scorecard():
    calc_colnames = ['ebitda_margin', 'ebitda_less_capex_margin', 'debt_to_ebitda', 'fcf_to_debt', 'ebit_to_interest_expense']
    moody_ratings = ['Aaa', 'Aa', 'A', 'Baa', 'Ba', 'B', 'Caa', 'Ca']
    scorecard = pd.DataFrame(columns = moody_ratings, index = ['revenue'] + calc_colnames)
    scorecard.loc['revenue', :]                     = [50, 30, 15, 5, 2, 0.75, 0.25, '<']   # ($, B)
    scorecard.loc['ebitda_margin', :]               = [0.5, 0.35, 0.3, 0.25, 0.2, 0.15, 0.10, '<']   # %
    scorecard.loc['ebitda_less_capex_margin', :]    = [0.35, 0.3, 0.25, 0.20, 0.15, 0.10, 0.05, '<']   # %
    scorecard.loc['debt_to_ebitda', :]              = [0.5, 1, 1.5, 2.5, 3.5, 5, 7, '>']   # 0x
    scorecard.loc['fcf_to_debt', :]                 = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, '<']   # %
    scorecard.loc['ebit_to_interest_expense', :]    = [30, 20, 10, 5, 3, 1.5, 0.0, '<']   # ($, B)
    moodys_weights = [0.2, 0.05, 0.05, 0.1, 0.1, 0.05] # buisness profile & financial policy are excluded as currently non-quantifiable
    scorecard['weights'] = moodys_weights
    print(scorecard)

# test_build_scorecard()