from tkinter import *
from tkinter import ttk

from view.widgets import Widgets

class MoodysMethodologies():

    def __init__(self, frame:ttk.Frame):
        self.root = frame

        self.top_frame =  ttk.Frame(self.root)
        self.top_frame.pack(side=TOP, expand=True, fill=BOTH)

        self.bottom_frame =  ttk.Frame(self.root)
        self.bottom_frame.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.draw_widgets()


    def draw_widgets(self):
        print('drawing widgets')

        Widgets().combobox( root  = self.top_frame, 
                            values = ["Semiconductor", "Information Technology", "Financials"], 
                            func = self.populate_table
                            ).pack(side=LEFT)


        Widgets().table(self.bottom_frame)


    def populate_table(self, sector:str):
        print(f'populating table with {sector}')
    
