from tkinter import *
from tkinter import ttk

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from view.widgets import Widgets

from model.attribution.Famma_French.famma_french import FammaFrench as FF


class Attribution:

    def __init__(self, frame:ttk.Frame):

        self.root = frame

        tabControl = ttk.Notebook(self.root)

        # Famma French
        self.tab1 = ttk.Frame(tabControl)
        tabControl.add(self.tab1, text ='Famma French')
        
        self.drop_down_frame =  ttk.Frame(self.tab1)
        self.drop_down_frame.pack(side=TOP, expand=True, fill=BOTH)

        self.graphics_table = ttk.Frame(self.tab1)  
        self.graphics_table.pack(side=BOTTOM,  expand=False)



        #####

        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab2, text ='tab2')

        tabControl.pack(expand = 1, fill ="both")


        # drop down selection frame
        self.drop_down_frame =  ttk.Frame(self.tab1)
        self.drop_down_frame.pack(side=TOP, anchor=W, expand=False)


        self.widgets = Widgets()

        self.draw_widgets()





    def plot(self):
        fig, ax = plt.subplots(1, 1)
        sns.set_style("ticks",{'axes.grid' : True})
        fig.suptitle(f'7 day change Distribution')
        ((self.df +1).cumprod()).plot(figsize=(15, 7))
        plt.title(f"Famma French Factors", fontsize=16)
        plt.ylabel('Portfolio Returns', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.graphics_table)  
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=BOTH)



    def run(self):
        ff = FF(symbols=['QQQ'], weights = [ 1.0 ])
        ff.merge_factors_and_portfolio(download_ff_data=False)
        self.df = ff.df

        ff.five_factor()
        model = ff.model

        # ff.print_summary()

        self.plot()


        

    def draw_widgets(self):
        ttk.Label(self.drop_down_frame, text='Enter {portfolio ticker : weights, ...').pack(side=LEFT)
        e = ttk.Entry( width=200)
        e.pack(side=LEFT)

        b = ttk.Button(text='Go', command=self.run())
        b.pack()



