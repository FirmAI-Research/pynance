import importlib

from subprocess import call
from tkinter import *
from tkinter import ttk

from view.widgets import Widgets
from model.equity.fundamentals import fundamentals

class Fundamentals():


    def __init__(self, frame:ttk.Frame):
        self.root = frame

        # drop down selection frame
        self.drop_down_frame =  ttk.Frame(self.root)
        self.drop_down_frame.pack(side=TOP, anchor=W, expand=False)

        # alt pandastable frame's
        self.alt_frame =  ttk.Frame(self.root)
        self.alt_frame.pack(side=TOP, expand=True, fill=X)

        self.side_table = ttk.Frame(self.alt_frame)  # Side half table for scorecard
        self.side_table.pack(side=RIGHT,  expand=True, fill=X )

        self.graphics_table = ttk.Frame(self.alt_frame)  # Side half table for scorecard
        self.graphics_table.pack(side=RIGHT,  expand=False, fill=Y)

        # main pandastable frame's
        self.main_table_frame =  ttk.Frame(self.root)
        self.main_table_frame.pack(side=BOTTOM, expand=False, fill=X)

        self.table_header = ttk.Frame(self.main_table_frame)
        self.table_header.pack(side=TOP, expand=False, fill=X)
 
        self.main_table = ttk.Frame(self.main_table_frame) # Main raw fundamentals data table
        self.main_table.pack(side=BOTTOM, expand=False, fill=X)

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
            print(f'Sector: {self.sector}, Industry: {self.industry}')

            # NOTE: use getattr to dynamicaly instantiate a class based on string selected in drop down
            try:        
                ttk.Label(self.table_header, text='All values reported in millions ($, M)').pack(side=BOTTOM, anchor = W)

                class_ = getattr(importlib.import_module("model.equity.fundamentals.fundamentals"), self.industry) #__import__('model.equity.fundamentals.fundamentals')
                instance = class_() # NOTE: instance here refers to an object of the fundamentals class with the name of the user's industry selection

                df_main = instance.build_table(self.sector, self.industry)
                Widgets().table(root = self.main_table, df = df_main, color_columns = instance.calc_colnames)

                df_side = instance.build_scorecard()
                Widgets().table(root = self.side_table, df = df_side) 

                Widgets().chart(root = self.graphics_table, df = df_side) 

            # NOTE: handle exceptions if user industry selection does not have a class associated with it
            except AttributeError:      
                print('[FAIL] Unable to populate table. Does a class exist for the selected industry in model/equity/fundamentals?')
                # TODO: populate a template/default table view...
