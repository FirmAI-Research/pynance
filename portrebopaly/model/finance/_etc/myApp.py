'''
michael
'''
import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
from tkinter.ttk import *
from tkinter import simpledialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,  NavigationToolbar2Tk


import numpy as np 

from OptionPayoff import OptionPayoff_Window
from BondPricing import BondPricing_Window

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self.master = master
        self.master = Tk() 
        self.master.geometry("1000x350") 
        self.create_widgets()
        self.master.mainloop()

    def create_widgets(self):
        option_list = [
            "Covered Call",
            "Straddle",
            "Strangle"
        ]
        strat = tk.StringVar(self.master, option_list)
        strat.set(option_list[0])
        def create_plot_window(selected_value):
            plot_window = OptionPayoff_Window(selected_value, self.master)
        opt = tk.OptionMenu(self.master, strat ,*option_list, command=create_plot_window)
        opt.pack(side="bottom")


        bond_list = [
            "Bond Price",
            "Yield Curve"
        ]
        bond_select = tk.StringVar(self.master, bond_list)
        bond_select.set(bond_list[0])
        def create_plot_window(selected_value):
            plot_window = BondPricing_Window(selected_value, self.master)
        bnd = tk.OptionMenu(self.master, bond_select ,*bond_list, command=create_plot_window)
        bnd.pack(side="bottom")



        exit_button = Button(self.master, text="Exit", command=self.master.quit) 
        exit_button.pack()


App()




'''
class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
'''