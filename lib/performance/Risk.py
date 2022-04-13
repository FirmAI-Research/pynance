TICKERS = ['AAPL','MSFT','BABA']
def get_cbind_onloop(tickers_list):
        df = yfin.download(tickers_list,'2015-1-1')['Adj Close']
        return df
cdf = get_cbind_onloop(TICKERS)


def risk_visuals(df):
    cdf_chg = df.pct_change()
    cdf_chg.tail()
    sns.jointplot(TICKERS[0],TICKERS[1],cdf,kind='scatter')
    sns.pairplot(cdf.dropna())
    corr = cdf.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, n =9, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


    plt.figure(figsize=(8,5))

    plt.scatter(cdf.mean(),cdf.std(),s=25)

    plt.xlabel('Expected Return')
    plt.ylabel('Risk')


    #For adding annotatios in the scatterplot
    for label,x,y in zip(cdf.columns,cdf.mean(),cdf.std()):
        plt.annotate(
        label,
        xy=(x,y),xytext=(-10,10),
        textcoords = 'offset points', ha = 'right', va = 'bottom',

    


risk_visuals(cdf)