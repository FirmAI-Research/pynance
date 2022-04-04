import importlib

from tkinter import *
from tkinter import ttk

from view.widgets import Widgets
from model.equity.insiders import Insiders as InsidersModel

class Insiders():

    def __init__(self, frame:ttk.Frame):
        self.root = frame
 
        # main pandastable frame's
        self.main_table_frame =  ttk.Frame(self.root)
        self.main_table_frame.pack(side=BOTTOM, expand=True, fill=BOTH)
 
        self.main_table = ttk.Frame(self.main_table_frame) # Main raw fundamentals data table
        self.main_table.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.widgets = Widgets()

        self.draw_widgets()

        self.core = InsidersModel()


    def draw_widgets(self):
        pass