from tkinter import *
from tkinter import ttk
import json
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

        self.tab1 = ttk.Frame(tabControl, relief=SUNKEN)
        tabControl.add(self.tab1, text ='Famma French')

        self.tab2 = ttk.Frame(tabControl)
        tabControl.add(self.tab2, text ='tab2')

        tabControl.pack(expand = 1,  fill = BOTH)

        # drop down selection frame
        self.drop_down_frame =  ttk.Frame(self.tab1)
        self.drop_down_frame.pack(side=TOP, expand=FALSE, fill=X)

        self.chart_frame =  ttk.Frame(self.tab1)
        self.chart_frame.pack(side=TOP, expand=True, fill=BOTH)

        
        self.model_frames_frame =  ttk.Frame(self.tab1)
        self.model_frames_frame.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.model_frame3 =  ttk.Frame(self.model_frames_frame)
        self.model_frame3.pack(side=LEFT, expand=True, fill=BOTH)

        self.model_frame4 =  ttk.Frame(self.model_frames_frame)
        self.model_frame4.pack(side=LEFT, expand=True, fill=BOTH)

        self.model_frame5 =  ttk.Frame(self.model_frames_frame)
        self.model_frame5.pack(side=LEFT, expand=True, fill=BOTH)

        self.widgets = Widgets()

        self.draw_widgets()


    def plot(self):
        fig, ax = plt.subplots(1, 1)
        sns.set_style("ticks",{'axes.grid' : True})
        chartdata = (self.df +1).cumprod()
        print(chartdata)
        sns.lineplot(data=chartdata)
        plt.title(f"Famma French Factors", fontsize=16)
        plt.ylabel('Portfolio Returns', fontsize=14)
        plt.xlabel('Year', fontsize=14)
        plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.chart_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=BOTH)


    def run(self):

        print(f'User selected text = {self.e.get()}')
        user_input_dict = json.loads(str(self.e.get()))
        print(user_input_dict.keys())
        print(user_input_dict.values())

        self.ff = FF(symbols=list(user_input_dict.keys()), weights = list(user_input_dict.values()))
        self.ff.merge_factors_and_portfolio(download_ff_data=False)
        self.df = self.ff.df

        self.ff.three_factor()
        self.model3 = self.ff.model

        self.ff.carhar_four_factor()
        self.model4 = self.ff.model

        self.ff.five_factor()
        self.model5 = self.ff.model
        
        self.plot()
        self.draw_widgets()


    def draw_widgets(self):

        # portfolio entry
        if not hasattr(self, 'e'):
            ttk.Label(self.drop_down_frame, text='Enter string dict() of {ticker : weights, ...} ').pack(side=LEFT)
            self.e = ttk.Entry( self.drop_down_frame, width=125)
            self.e.pack(side=LEFT, anchor=N)
            b = ttk.Button(self.drop_down_frame, text='Go', command= lambda : self.run())
            b.pack(side=LEFT, anchor=N)


        # statsmodels display
        if hasattr(self, 'ff'):
            # 3 factor
            ttk.Label(self.model_frame3, text='3 factor').pack(side=TOP)
            sns.set_style("dark")
            fig = plt.figure()
            plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.text(0.0, 0.0, str(self.model3.summary()), {'fontsize': 9}, fontproperties = 'monospace')  
            canvas = FigureCanvasTkAgg(fig, master = self.model_frame3)  
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill=BOTH)
            
            # 4 factor
            ttk.Label(self.model_frame4, text='4 factor').pack(side=TOP)
            sns.set_style("dark")
            fig = plt.figure()
            plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.text(0.0, 0.0, str(self.model4.summary()), {'fontsize': 9}, fontproperties = 'monospace')  
            canvas = FigureCanvasTkAgg(fig, master = self.model_frame4)  
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill=BOTH)

            # 5 factor
            ttk.Label(self.model_frame5, text='5 factor').pack(side=TOP)
            sns.set_style("dark")
            fig = plt.figure()
            plt.axis('off')
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.text(0.0, 0.0, str(self.model5.summary()), {'fontsize': 9}, fontproperties = 'monospace')  
            canvas = FigureCanvasTkAgg(fig, master = self.model_frame5)  
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill=BOTH)