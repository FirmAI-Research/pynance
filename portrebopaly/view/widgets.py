import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledText
# from pandastable import Table, TableModel
from ttkbootstrap.tableview import Tableview

from tkinter.ttk import *

from tkinter import *

class Widgets():

    def __init__(self):
        pass

    def combobox(self, root:ttk.Frame, values:list):
        cb = ttk.Combobox(root, state='readonly', values=values)
        cb.current()
        cb.bind('<<ComboboxSelected>>', lambda event: print(event.widget.get()))
        return cb


    def table(self, app):
        app = ttk.Toplevel()
        #colors = app.style.colors

        coldata = [
            {"text": "LicenseNumber", "stretch": False},
            "CompanyName",
            {"text": "UserCount", "stretch": False},
        ]

        rowdata = [
            ('A123', 'IzzyCo', 12),
            ('A136', 'Kimdee Inc.', 45),
            ('A158', 'Farmadding Co.', 36)
        ]

        dt = Tableview(
            master=app,
            coldata=coldata,
            rowdata=rowdata,
            paginated=True,
            searchable=True,
            bootstyle=PRIMARY,
            #stripecolor=(colors.light, None),
        )
        dt.pack(fill=BOTH, expand=YES, padx=10, pady=10)


    def pt_table(self, root:ttk.Frame):
        df = TableModel.getSampleData()
        self.table = pt = Table(root, dataframe=df,
                                showtoolbar=True, showstatusbar=True)
        ttk.Style().configure("TButton", padding=6, relief="flat", background="#ccc")
        self.table.show()