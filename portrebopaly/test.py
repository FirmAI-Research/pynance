
import pandas as pd
import nasdaqdatalink
from controller.calendar import Calendar
from vendors.nasdaq import Nasdaq, CoreUsFundamentals,  CoreUSInstitutionalInvestors,  Tickers, Insiders
import matplotlib.pyplot as plt
import seaborn as sns 
from model.attribution.Famma_French.famma_french import FammaFrench

# -- NasdaqDataLink Api Call --
def test_api_call():
    core = CoreUsFundamentals()
    df = core.get()
    print(df)
# test_api_call()

# -- CoreUsFundamentals sample export -- 
def test_core_us_fundamentals():
    core = CoreUsFundamentals()
    # df_core =  core.get_export()
    df_core = core.get()
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
def test_institutions_groups():
    core = CoreUSInstitutionalInvestors()
    df = core.get()
    print(core.get())
    print(df.calendardate.value_counts())
    print(df.columns)
    print(df.tail())
    print(df.shape)
    print(core.group_by_ticker(df))
    core.group_by_institution()


# -- CoreUSInstitutionalInvestors plot -- 
def test_institutions_plot():
    core = CoreUSInstitutionalInvestors()

    df_qe = core.get_export(fp = './vendors/exports/SHARADAR_SF3_Full_Export.csv')
    df_prior = core.get_export(fp = './vendors/exports/SHARADAR_SF3_Prior_Qtr.csv')

    institution='BRIDGEWATER ASSOCIATES LP'
    df = core.time_series_range(institution=institution,  qtr_start='2021-01-01', qtr_end='2021-12-31')
    df.to_csv(f'./vendors/output/{institution}.csv')
    df.calendardate = pd.to_datetime(df.calendardate)
    print(df.head())

    # plot
    fig, axes = plt.subplots(1, 1, figsize=(15,8))
    line = sns.lineplot(data=df, x="calendardate", y="value", hue='ticker')
    annotate_txt_df = df.loc[df.calendardate=='2020-12-31'].reset_index()
    print(annotate_txt_df)
    for i in range(len(annotate_txt_df.ticker)):
        line.annotate(annotate_txt_df.ticker[i], xy = (annotate_txt_df.calendardate[i], annotate_txt_df.value[i]) )
    plt.legend(loc='upper right')
    plt.show()
    ts = core.time_series_range(institution, qtr_start='2020-12-31', qtr_end='2021-12-31')
    print(ts)
    df_increase, df_decrease = core.change()
    print(df_increase)
    print(df_decrease)


# -- CoreUSInstitutionalInvestors better api calls -- 
def test_institutions_api_call():
    core = CoreUSInstitutionalInvestors()

# test_institutions_api_call()



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



def test_famma_french():
    ff = FammaFrench(symbols=['QQQ'], weights = [ 1.0 ])
    ff.merge_factors_and_portfolio(download_ff_data=False)
    ff.five_factor()
    ff.print_summary()
    ff.plot()
# test_famma_french()


def test_insiders():
    from model.equity.insiders.insiders import Insiders as InsidersModel

    ins = Insiders()
    ins.curl()
    # ins.direct_by_ticker()
# test_insiders()


def timeseries():
    from model.time_series import arima

timeseries()
