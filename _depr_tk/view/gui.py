from ttkthemes import ThemedTk

from tkinter import *
from tkinter import ttk

from view.fundamentals import Fundamentals
from view.institution import Institution
from view.treasury_rates import TreasuryRates
from view.insiders import Insiders
from view.attribution import Attribution
from view.timeseries import TimeSeries

from view.menubar import MenuBar
from view.widgets import Widgets


class Gui():

    def __init__(self):    

        root = ThemedTk(theme="breeze")

        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth()-3, root.winfo_screenheight()-3))
        root.title("PortRebOpAly")
        tabControl = ttk.Notebook(root)
        
        self.tab_fundamentals = ttk.Frame(tabControl)
        self.tab_institutions = ttk.Frame(tabControl)
        self.tab_interest_rates = ttk.Frame(tabControl)
        self.tab_time_series = ttk.Frame(tabControl)
        self.tab_attribution = ttk.Frame(tabControl)

        self.insiders = ttk.Frame(tabControl)


        self.tab_generator = ttk.Frame(tabControl)

        MenuBar(root)

        # tabs
        tabControl.add(self.tab_fundamentals, text ='Fundamentals')
        tabControl.add(self.tab_institutions, text ='Institutions')
        tabControl.add(self.tab_interest_rates, text ='Treasury Rates')
        tabControl.add(self.tab_time_series, text ='Time Series')
        tabControl.add(self.tab_attribution, text ='Attribution')
        tabControl.add(self.insiders, text ='Insiders')
        tabControl.add(self.tab_generator, text ='Report Generator')
        tabControl.pack(expand = 1, fill ="both")

        # widgets
        Fundamentals(frame = self.tab_fundamentals)
        Institution(frame = self.tab_institutions)
        TreasuryRates(frame = self.tab_interest_rates)
        Attribution(frame = self.tab_attribution)
        TimeSeries(frame = self.tab_time_series)
        # Insiders(frame = self.insiders)

        root.mainloop()


