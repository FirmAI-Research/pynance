
import importlib

# from subprocess import call
from tkinter import *
from tkinter import ttk
import seaborn as sns

from view.widgets import Widgets
from vendors.nasdaq import Nasdaq, CoreUSInstitutionalInvestors,  Tickers

class Institution:

    def __init__(self, frame:ttk.Frame):
        self.root = frame

        # main pandastable frame's
        self.main_table_frame =  ttk.Frame(self.root)
        self.main_table_frame.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.tickers_table_head = ttk.Frame(self.main_table_frame)
        self.tickers_table_head.pack(side=TOP, expand=False, fill=X)

        self.tickers_table = ttk.Frame(self.main_table_frame) # Main raw fundamentals data table
        self.tickers_table.pack(side=LEFT, expand=True, fill=BOTH)

        ##
        self.institution_table = ttk.Frame(self.main_table_frame) # Main raw fundamentals data table
        self.institution_table.pack(side=RIGHT, expand=True, fill=BOTH)

        self.institution_table_head = ttk.Frame(self.institution_table) # Main raw fundamentals data table
        self.institution_table_head.pack(side=TOP, expand=False, fill=X)


        self.institution_table_data = ttk.Frame(self.institution_table) # Main raw fundamentals data table
        self.institution_table_data.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.widgets = Widgets()

        ttk.Label(self.institution_table_head, text='Institution').pack(side=TOP)
        ttk.Label(self.tickers_table_head, text='All values reported in billions ($, B)').pack(side=BOTTOM, anchor = W)

        self.core = CoreUSInstitutionalInvestors()

        ttk.Button(self.tickers_table_head, text = 'Time Series', command = lambda : self.quarterly_change() ).pack(side=TOP, anchor = E)

        self.draw_widgets()


    def draw_widgets(self):
        core = CoreUSInstitutionalInvestors()

        self.display_table()


    def _setattr(self, event_widget, call_name, selection):
        """ sets class attributes to the instance of this (self) class 
        """
        setattr(self, call_name, selection)

        self.draw_widgets()


    def display_table(self):
     
        self.core.get_export()  # sets core.df class attribute

        # by Ticker
        df_tickers = self.core.group_by_ticker().reset_index()
        Widgets().table(root = self.tickers_table, df = df_tickers, color_columns = None)

        # by Institution
        if not hasattr(self, 'institution'): # dont draw again if box 
            ttk.Label(self.institution_table_head, text='All values reported in millions ($, M)').pack(side=BOTTOM, anchor = W)
            cb = self.widgets.combobox( root  = self.institution_table_head, 
                                values = self.core.list_all_institutions(), 
                                func = self._setattr,
                                call_name = 'institution',
                                state = 'enable')
            cb.pack(side=TOP)

        if hasattr(self, 'institution'):

            df_institutions = self.core.group_by_institution().reset_index()
            df_filter =  df_institutions.loc[df_institutions['investorname'] == self.institution]
            Widgets().table(root = self.institution_table_data, df = df_filter, color_columns = None)



    def quarterly_change(self):
        window =  self.widgets.new_window(self.root)
        df = self.core.qtr_over_qtr_change(self.institution, qtr_start='2020-12-31', qtr_end='2021-12-31')
        institution = self.institution
        Widgets().single_chart(root = window, df = df, x='calendardate', y='value', h='ticker', chart_type = sns.lineplot, title = institution)

        