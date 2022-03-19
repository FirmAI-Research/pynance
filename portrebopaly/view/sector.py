from tkinter import *
from tkinter import ttk
from pandastable import Table, TableModel

class Sector():

    def __init__(self, root:ttk.Frame):
        self.root = root

        self.top_frame =  ttk.Frame(self.root, )
        self.top_frame.pack(side=TOP, expand=True, fill=BOTH)

        self.bottom_frame =  ttk.Frame(self.root)
        self.bottom_frame.pack(side=BOTTOM, expand=True, fill=BOTH)


        self.draw_widgets()

    def draw_widgets(self):
        ttk.Button(self.top_frame, text ="Moody's Sector Methodologies").grid(column = 0, row = 1,  padx = 30,  pady = 30)

        df = TableModel.getSampleData()
        self.table = pt = Table(self.bottom_frame, dataframe=df, showtoolbar=True, showstatusbar=True)
        self.table.show()