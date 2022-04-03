from tkinter import *
from tkinter import ttk

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from view.widgets import Widgets

from model.fixed_income.rates.treasury_rates import TreasuryRates as TR


class TreasuryRates:

    def __init__(self, frame:ttk.Frame):
        self.root = frame


        self.by_time_frame =  ttk.Frame(self.root)
        self.by_time_frame.pack(side=LEFT,expand=False)

        self.by_tenor_frame =  ttk.Frame(self.root)
        self.by_tenor_frame.pack(side=RIGHT,expand=True, fill=BOTH)

        self.yields_over_time_frame =  ttk.Frame(self.by_time_frame)
        self.yields_over_time_frame.pack(side=TOP,expand=False)

        self.recent_yields_over_time_frame =  ttk.Frame(self.by_time_frame)
        self.recent_yields_over_time_frame.pack(side=BOTTOM,expand=False)

        self.yield_curve_frame =  ttk.Frame(self.by_tenor_frame)
        self.yield_curve_frame.pack(side=TOP,expand=True)

        self.points_in_time_frame =  ttk.Frame(self.by_tenor_frame)
        self.points_in_time_frame.pack(side=BOTTOM,  expand=True)

        self.widgets = Widgets()

        self.tr = TR()

        self.draw_widgets()



    def draw_widgets(self):
        
        # Yield Curve of most recent data
        df = self.tr.plot_curve()
        fig, ax = plt.subplots(1, 1, figsize=(11,6))
        fig.suptitle(f'Yield Curve as of Yesterday')
        ax = sns.lineplot(x = 'variable', y = 'value', data = df)
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.yield_curve_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack()


        # Yield Curve at points in time
        df = self.tr.plot_curve_at_points_in_time()
        fig, ax = plt.subplots(1, 1, figsize=(11,6))
        fig.suptitle('Yields at points in time')
        ax = sns.lineplot(x = 'variable', y = 'value', hue = 'date', data = df)
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.points_in_time_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack()

        # All tenors over time
        df = self.tr.ts_by_years(years = ['2022', '2021', '2020','2019'])
        fig, ax = plt.subplots(1, 1, figsize=(11,6))
        fig.suptitle('All tenors over time')
        ax = sns.lineplot(x = 'date', y = 'value', hue = 'variable', data = df)
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.yields_over_time_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Yield Curve of most recent data
        n = 14
        df = self.tr.ts_by_months(years = ['2022'], n=n)
        fig, ax = plt.subplots(1, 1, figsize=(11,6))
        fig.suptitle(f'All tenors last {n} days')
        ax = sns.lineplot(x = 'date', y = 'value', hue = 'variable', data = df)
        plt.setp(ax.get_xticklabels(), rotation=20)
        canvas = FigureCanvasTkAgg(fig, master = self.recent_yields_over_time_frame)  
        canvas.draw()
        canvas.get_tk_widget().pack()
