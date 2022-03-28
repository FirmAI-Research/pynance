from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import pandastable
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


    def table(self, root:ttk.Frame, df:pd.DataFrame, color_columns:list=None):
        pd.options.display.float_format = '{:,.2f}'.format
        print(df)
        
        table = pt = Table(root, dataframe=df,showtoolbar=True, showstatusbar=True)
        table.show()

        options = {'align': 'w',
        'cellbackgr': '#F4F4F3',
        'cellwidth': 80,
        'colheadercolor': '#535b71',
        'floatprecision': 1,
        'font': 'Arial',
        'fontsize': 12,
        'fontstyle': '',
        'grid_color': '#ABB1AD',
        'linewidth': 1,
        'rowheight': 22,
        'rowselectedcolor': '#E4DED4',
        'textcolor': 'black'}
        pandastable.config.apply_options(options, table)

        # color a subset of columns
        if color_columns is not None:
            for c in color_columns:
                table.columncolors[c] = '#dcf1fc' #color a specific column

        table.redraw()

    
    def chart(self,  root:ttk.Frame, df:pd.DataFrame,):
        fig = Figure(figsize = (5, 5),
                    dpi = 100)
    
        y = [i**2 for i in range(101)]
    
        plot1 = fig.add_subplot(111)
        plot1.plot(y)
        canvas = FigureCanvasTkAgg(fig, master = root)  
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack()