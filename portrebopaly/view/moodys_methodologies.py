import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from view.widgets import Widgets

class MoodysMethodologies():

    def __init__(self, root, frame:ttk.Frame):
        self.app = root
        self.root = frame

        self.top_frame =  ttk.Frame(self.root)
        self.top_frame.pack(side=TOP, expand=True, fill=BOTH)

        self.bottom_frame =  ttk.Frame(self.root)
        self.bottom_frame.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.draw_widgets()


    def draw_widgets(self):

        # Widgets().combobox(self.top_frame, ["Semiconductor", "Information Technology", "Financials"]).pack(side=LEFT)

        # Widgets().pt_table(self.bottom_frame)

        Widgets().table(self.app)
