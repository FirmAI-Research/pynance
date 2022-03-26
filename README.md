# portrebopaly
*portfolio rebalancing, optimization, and analysis*

vendors:
* yfinance                          |   https://github.com/ranaroussi/yfinance
* Sharadar Core US Equities Bundle  |   https://data.nasdaq.com/databases/SFA/data  |   nasdaq data link (quandl)
* yahoo_fin                         |   http://theautomatic.net/yahoo_fin-documentation/
* fred api                          |   https://github.com/mortada/fredapi

</br>

*collection* --> aws ec2 | lambda functions --> daily pulls from above vendors  </br>
*storage* --> aws rds postgreesql  </br></br>


<strong>portrebopaly/</strong>


model/

    attribution/
        risk & return performance attribution

    backtest/
        equity investment strategy backtesting 
    
    workflow/
        integrated workflow of various analytical tools included in the project for different products (i.e. equity or fixed income securities)

    finance/
        pricing fixed income instruments

    optimization/
        portfolio optimization using my forked version of PyPortfolioOpt

    performance/
        calculating performance
    
    rebalance/
        inteligent portfolio rebalancing 

    sec_nlp/
        summarization and key word extraction from sec corporate filings
    
    time_series/
        time series analysis

    flows/
        institutional v retail activity

    options/
        implied volitilities & put to call ratio's

    moodys_methodologies/




view/

    gui/

        interface has four "starting points"

        1. Single Stock Analysis    [Security]
            * workflows
            * technicals
            * fundamentals
            * sec filing nlp
            * dcf
        2. Portfolio Analysis   [Portfolio]
            * optimization
            * rebalance
        3. Market [Market]
            * screens
            * backtests
        4. Sector [Sector]
            * screens
            * backtests




controller/











NasdaqDataLink

