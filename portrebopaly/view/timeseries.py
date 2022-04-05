
from tkinter import *
from tkinter import ttk

from view.widgets import Widgets


class TimeSeries:


    def __init__(self, frame:ttk.Frame):

        self.root = frame


        tabControl = ttk.Notebook(self.root)

        self.tab1 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text ='tab1')

        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab2, text ='tab2')

        tabControl.pack(expand = 1, fill ="both")


        # drop down selection frame
        self.drop_down_frame =  ttk.Frame(self.tab1)
        self.drop_down_frame.pack(side=TOP, anchor=W, expand=False)


        self.widgets = Widgets()

        self.draw_widgets()



    def draw_widgets(self):

        # ticker dropdown selection
        if not hasattr(self, 'ticker'):
            ttk.Label(self.drop_down_frame, text='ticker').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.drop_down_frame, 
                                values = ['AAPL', 'MSFT', 'GOOG', 'AMZN'], 
                                func = self._setattr,
                                call_name = 'ticker')
            cb.pack(side=LEFT)


        # industry dropdown selection
        if hasattr(self, 'ticker') and not hasattr(self, 'model'):
            ttk.Label(self.drop_down_frame, text='model').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.drop_down_frame, 
                                values = ['arima','sarimax','garch'], 
                                func = self._setattr,
                                call_name = 'model')
            cb.pack(side=LEFT)
            
        # display table
        if hasattr(self, 'ticker') and hasattr(self, 'model'):
            self.display_table()
            pass


    def _setattr(self, event_widget, call_name, selection):
        """ sets class attributes to the instance of this (self) class 
        """
        setattr(self, call_name, selection)

        back_selection = self.widgets.check_for_back_selection(self.drop_down_frame, event_widget)

        if back_selection == True:
            try:
                delattr(self, 'model')
            except:
                pass
            
        self.draw_widgets()