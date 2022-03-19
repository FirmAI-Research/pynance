import QuantLib as ql 

usd = ql.USDCurrency()
eur = ql.EURCurrency()
usdToeur = ql.ExchangeRate(eur, usd, 1.14)
m_usd = 5 * usd
m_eur = 4.39 * eur
print( 'Converting from USD: ', m_usd, ' = ', usdToeur.exchange(m_usd))
print( 'Converting from EUR: ', m_eur, ' = ', usdToeur.exchange(m_eur))

print(usdToeur.source())
print(usdToeur.target())
print(usdToeur.rate())


