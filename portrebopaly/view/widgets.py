from tkinter import *
from tkinter import ttk

from pandastable import Table, TableModel
import pandas as pd

class Widgets():

    def __init__(self):
        pass



    def check_for_back_selection(self, frame, event_widget):
        """ check if widget event occured on a combobox widget that was not the last combobox widget to be packed to a frame
        
        """
        widgets_on_window = [entry for entry in frame.winfo_children()]
        event_widget_index = [i  for i, e in enumerate(widgets_on_window) if e.winfo_name() == event_widget.winfo_name()][0]
        if event_widget.winfo_name() == widgets_on_window[-1].winfo_name():
            return False
        else:
            for ix, widget in enumerate(widgets_on_window):
                if ix > event_widget_index:
                    frame.nametowidget(widget).pack_forget()                 # remove widgets were packed before the event widget
            return True


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
        cb.bind('<<ComboboxSelected>>', lambda event: func(cb, call_name, event.widget.get()))
        return cb


    def table(self, root:ttk.Frame, df:pd.DataFrame):
        # df = TableModel.getSampleData()
        table = pt = Table(root, dataframe=df,showtoolbar=True, showstatusbar=True)
        table.show()
        table.redraw()