import datetime
import dateutil.rrule as rrule

class Calendar:

    def __init__(self):
        pass


    def today(self):
        return datetime.datetime.today()#.strftime('%Y-%m-%d')

    
    def previous_quarter_end(self, dt_obj = datetime.datetime.today()):
        rr = rrule.rrule(rrule.DAILY,bymonth=(3,6,9,12), bymonthday=-1, dtstart = dt_obj-datetime.timedelta(days=100))
        result = rr.before(dt_obj, inc=False) 
        return result.date()