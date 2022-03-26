from tkinter import *
from tkinter import ttk

from pandastable import Table, TableModel
import pandas as pd

class Widgets():

    def __init__(self):
        pass


    def reset(self,cb):
        """ reset combobox selections
        """
        pass


    def combobox(self, root:ttk.Frame, values:list, func=None, call_name=None):
        """ create a dropdown menu

        :param root: Frame for table to appear on
        :param values: list of sting values to populate combobox options with
        :param func: a function object to be called when combobox option is selected.
            the function should take one argument, which is the selected string option value
        :param call_name: a string used to refer to the name of the interaction that created this object
            used when setting class attributes

        :return: combobox object
            pack() or grid() the combobox using the returned object
        """
        cb = ttk.Combobox(root, state='readonly', values=values)
        cb.current()
        cb.bind('<<ComboboxSelected>>', lambda event: func(call_name, event.widget.get()))
        return cb


    def table(self, root:ttk.Frame, df:pd.DataFrame):
        # df = TableModel.getSampleData()
        table = pt = Table(root, dataframe=df,showtoolbar=True, showstatusbar=True)
        table.show()
        table.redraw()