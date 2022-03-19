import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
from tkinter.ttk import *
from tkinter import simpledialog
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,  NavigationToolbar2Tk
from options import covered_call as  covered_call_payoff
from options import strangle as strangle_payoff
from options import straddle as straddle_payoff
import numpy as np 


class OptionPayoff_Window(tk.Frame):

    def __init__(self, window_type, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.window_type = window_type
 
        self.openwindow()
        self.create_widgets()
        self.x = 'michael sands'

    def openwindow(self): 
        self.window = Toplevel(self.master) 
        self.window.title("New Window") 
        self.window.geometry("1000x350") 
        Label(self.window,  text ="This is a new window").pack() 

        self.create_widgets_child_window()

    def plot(self):
        fig = Figure(figsize = (5, 5), dpi = 100) 
        plot1 = fig.add_subplot(111) 

        plot1.plot(self.sT,self.y1,lw=1.5,label='Long Stock')
        plot1.plot(self.sT,self.y2,lw=1.5,label='Short Call')
        plot1.plot(self.sT,self.y3,lw=1.5,label='Covered Call')

        plot1.grid(True)
        plot1.axis('tight')
        plot1.legend(loc=0)

        canvas = FigureCanvasTkAgg(fig, master = self.window)   
        canvas.draw()
        ll = Label(master=self.window, text=self.payoff_description_str)
        ll.pack(side="top")
        canvas.get_tk_widget().pack() 
        toolbar = NavigationToolbar2Tk(canvas, self.window) 
        toolbar.update()     
        canvas.get_tk_widget().pack() 

    def create_widgets_child_window(self):

        '''collect input params here on window pre plot click'''

        if self.window_type =='Covered Call':
            self.s0 = 100 # initial underlying price (spot price)
            self.k = 110 # strike price
            self.c = 1.35 # premium paid for contract
            self.nshares = 100 # number of shares per lot
            self.sT = np.arange(0,2*self.s0,5)# index array of stock prices over time T
            self.payoff_description_str = f'{self.c} premium paid for {self.k} strike for {self.window_type} on  underlying of {self.s0} over {self.nshares}'
            self.days_to_expiration = None
            cc = covered_call_payoff.calc_payoff(self.s0, self.k, self.c, self.nshares, self.sT)
            self.y1,self.y2,self.y3 = cc[0],cc[1],cc[2]

        if self.window_type in  ['Strangle', 'Straddle']:   

            if self.window_type =='Strangle':
                self.s0 = 100
                self.c_long_put = 1.35
                self.k_long_put = 100
                self.c_long_call = 1.25
                self.k_long_call =  124
                self.sT = np.arange(0,2*self.s0,5)
                cc =  strangle_payoff.calc_payoff(self.s0, self.c_long_put, self.k_long_put,  self.c_long_call, self.c_long_put, self.sT)
                self.y1,self.y2,self.y3,ml = cc[0],cc[1],cc[2],cc[3]
            if self.window_type == 'Straddle':
                self.s0 = 100
                self.c_long_put = 1.35
                self.k_long_put = 100
                self.c_long_call = 1.25
                self.k_long_call =  100
                self.sT = np.arange(0,2*self.s0,5)
                cc =  straddle_payoff.calc_payoff(self.s0, self.c_long_put, self.k_long_put,  self.c_long_call, self.c_long_put, self.sT)
                self.y1,self.y2,self.y3 = cc[0],cc[1],cc[2]

            self.payoff_description_str = f'Strategy: {self.window_type}'
            self.days_to_expiration = None

        plot_button = Button(master = self.window, 
                            text = "Plot",
                            command=self.plot) 
        plot_button.pack() 
        
    def create_widgets(self):
        exit_button = Button(self.window, text="Exit", command=self.window.destroy) 
        exit_button.pack()
        # self.window.wait_window(self.window)        


