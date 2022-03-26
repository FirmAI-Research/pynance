from subprocess import call
from tkinter import *
from tkinter import ttk

from view.widgets import Widgets

from model.equity.fundamentals import fundamentals

class MoodysMethodologies():


    def __init__(self, frame:ttk.Frame):
        self.root = frame

        self.top_frame =  ttk.Frame(self.root)
        self.top_frame.pack(side=TOP, expand=True, fill=BOTH)

        self.bottom_frame =  ttk.Frame(self.root)
        self.bottom_frame.pack(side=BOTTOM, expand=False, fill=X)

        self.widgets = Widgets()

        self.draw_widgets()


    def draw_widgets(self):

        # sector
        if hasattr(self, 'widgets') and not hasattr(self, 'sector'):
            ttk.Label(self.top_frame, text='Sector').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.top_frame, 
                                values = ["Technology", "Financials"], 
                                func = self._setattr,
                                call_name = 'sector')
            cb.pack(side=LEFT)

        # industry
        if hasattr(self, 'sector') and not hasattr(self, 'industry'):
            ttk.Label(self.top_frame, text='Industry').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.top_frame, 
                                values = ["Semiconductor", "Insurance", "Banks"], 
                                func = self._setattr,
                                call_name = 'industry')
            cb.pack(side=LEFT)
        

    def _setattr(self, call_name, selection):
        """ sets class attributes to the instance of this class 
        """
        setattr(self, call_name, selection)
        self.draw_widgets()
        self.display_table()


    def display_table(self):

        if hasattr(self, 'sector') and hasattr(self, 'industry'):

            self.df = fundamentals.filter_by_sector(self.sector)

            Widgets().table(root = self.bottom_frame, df = self.df)
