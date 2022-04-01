
import pandas as pd
import nasdaqdatalink
from controller.calendar import Calendar
from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  CoreUSInstitutionalInvestors,  Tickers
import matplotlib.pyplot as plt
import seaborn as sns 

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
    # print(core.get())
    # print(df.calendardate.value_counts())
    # print(df.columns)
    # print(raw_df.tail())
    # print(raw_df.shape)
    # print(core.group_by_ticker(raw_df))
    # core.group_by_institution()

    # df_qe = core.get_export(fp = './vendors/exports/SHARADAR_SF3_Full_Export.csv')
    # df_prior = core.get_export(fp = './vendors/exports/SHARADAR_SF3_Prior_Qtr.csv')

    # quarter over quarter
    # dates =  Calendar().quarter_end_list('2020-12-31', '2021-12-31')
    # frames = []
    # for date in dates:
    #     df = core.get(date = date, institution='BLACKROCK INC')
    #     df = df.sort_values(by=['value'], ascending=False)
    #     df = df.iloc[:20]
    #     frames.append(df)
    #     print(df.head())
    #     print(df.shape)
    # df = pd.concat(frames)
    # df.to_csv('./vendors/output/blackrock.csv')
    df = pd.read_csv('./vendors/output/blackrock.csv')
    df.calendardate = pd.to_datetime(df.calendardate)
    print(df.head())
    sns.lineplot(data=df, x="calendardate", y="value", hue='ticker')
    plt.show()




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