
from pandastable import Table, TableModel


from tkinter import *
from tkinter import ttk


class Widgets():

    def __init__(self):
        pass

    def combobox(self, root:ttk.Frame, values:list):
        cb = ttk.Combobox(root, state='readonly', values=values)
        cb.current()
        cb.bind('<<ComboboxSelected>>', lambda event: print(event.widget.get()))
        return cb


    def pt_table(self, root:ttk.Frame):
        df = TableModel.getSampleData()
        self.table = pt = Table(root, dataframe=df,showtoolbar=True, showstatusbar=True)
        self.table.show()
        self.table.redraw()