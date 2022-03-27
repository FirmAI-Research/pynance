import importlib

from subprocess import call
from tkinter import *
from tkinter import ttk

from view.widgets import Widgets
from model.equity.fundamentals import fundamentals

class Fundamentals():


    def __init__(self, frame:ttk.Frame):
        self.root = frame

        self.drop_down_frame =  ttk.Frame(self.root)
        self.drop_down_frame.pack(side=TOP, expand=False, fill=X)

        self.table_frame =  ttk.Frame(self.root)
        self.table_frame.pack(side=BOTTOM, expand=False, fill=X)

        self.widgets = Widgets()

        self.draw_widgets()


    def draw_widgets(self):

        # sector dropdown selection
        if not hasattr(self, 'sector'):
            ttk.Label(self.drop_down_frame, text='Sector').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.drop_down_frame, 
                                values = fundamentals.list_sectors(), 
                                func = self._setattr,
                                call_name = 'sector')
            cb.pack(side=LEFT)

        # industry dropdown selection
        if hasattr(self, 'sector') and not hasattr(self, 'industry'):
            ttk.Label(self.drop_down_frame, text='Industry').pack(side=LEFT)
            cb = self.widgets.combobox( root  = self.drop_down_frame, 
                                values = fundamentals.list_industries(self.sector), 
                                func = self._setattr,
                                call_name = 'industry')
            cb.pack(side=LEFT)
            
        # display table
        if hasattr(self, 'sector') and hasattr(self, 'industry'):
            self.display_table()
            pass


    def _setattr(self, event_widget, call_name, selection):
        """ sets class attributes to the instance of this (self) class 
        """
        setattr(self, call_name, selection)

        back_selection = self.widgets.check_for_back_selection(self.drop_down_frame, event_widget)

        if back_selection == True:
            try:
                delattr(self, 'industry')
            except:
                pass
            
        self.draw_widgets()


    def display_table(self):

        if hasattr(self, 'sector') and hasattr(self, 'industry'):

            # NOTE: use getattr to dynamicaly instantiate a class based on string selected in drop down
            try:        
                class_ = getattr(importlib.import_module("model.equity.fundamentals.fundamentals"), self.industry) #__import__('model.equity.fundamentals.fundamentals')
                instance = class_() # NOTE: instance here refers to an object of the fundamentals class with the name of the user's industry selection
                print(instance.colnames)
                df = instance.build_table(self.sector, self.industry)
                Widgets().table(root = self.table_frame, df = df)
            
            # NOTE: handle exceptions if user industry selection does not have a class associated with it
            except:      
                print('[FAIL] Unable to populate table. Does a fundamental class exist for the selected industry?')
                # TODO: populate a template/default table view...
