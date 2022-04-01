import datetime
import dateutil.rrule as rrule
import pandas as pd 


class Calendar:

    def __init__(self):
        pass


    def today(self):
        return datetime.datetime.today()#.strftime('%Y-%m-%d')


    def previous_quarter_end(self, dt_obj = datetime.datetime.today()):
        rr = rrule.rrule(rrule.DAILY,bymonth=(3,6,9,12), bymonthday=-1, dtstart = dt_obj-datetime.timedelta(days=100))
        result = rr.before(dt_obj, inc=False) 
        return result.date()

    def prior_quarter_end(self):
        return self.previous_quarter_end() - datetime.timedelta(months=3)

    
    def quarter_end_list(self, start_date, end_date):
        return pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.offsets.QuarterBegin(1), freq='Q').strftime('%Y-%m-%d').tolist()