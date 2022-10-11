import datetime
import dateutil.rrule as rrule
import pandas as pd 
from dateutil.relativedelta import relativedelta

import pandas_market_calendars as mcal
nyse = mcal.get_calendar('NYSE')


class Calendar:

    def __init__(self):
        pass


    def today(self):
        return datetime.datetime.today()
        
    def current_year(self, dtobj = datetime.datetime.today()):
        return dtobj.year


    def previous_month_end(self, offset:int = None):
        first = self.today().replace(day=1)
        last_month_end = first - datetime.timedelta(days=1)
        if offset == None:
            return last_month_end
        else:
            return last_month_end + relativedelta(months=offset)


    def closest_market_day(self, dtobj):
        market_days = nyse.valid_days(start_date = '1900-01-01', end_date = '2100-01-01')
        date_diff = { abs(dtobj.timestamp() - date.timestamp()) : date for date in market_days}
        return date_diff[min(date_diff.keys())]

        
    def previous_quarter_end(self, dt_obj = datetime.datetime.today()):
        rr = rrule.rrule(rrule.DAILY,bymonth=(3,6,9,12), bymonthday=-1, dtstart = dt_obj-datetime.timedelta(days=100))
        result = rr.before(dt_obj, inc=False) 
        return result.date()


    def prior_quarter_end(self):
        return self.previous_quarter_end() - relativedelta(months=3)

    
    def quarter_end_list(self, start_date, end_date):
        return pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.offsets.QuarterBegin(1), freq='Q').strftime('%Y-%m-%d').tolist()

    