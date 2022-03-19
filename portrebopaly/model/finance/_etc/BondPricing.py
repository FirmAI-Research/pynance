import tkinter as tk
from tkinter import *
from tkinter import messagebox as mb
from tkinter.ttk import *

class BondPricing_Window(tk.Frame):
    def __init__(self, window_type, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.window_type = window_type
 
        self.openwindow()
        self.create_widgets()
        self.x = 'michael sands'


    def create_widgets_child_window(self):

        plot_button = Button(master = self.window, 
                            text = "Plot",
                            command=self.plot) 
        plot_button.pack() 


    def openwindow(self): 
        self.window = Toplevel(self.master) 
        self.window.title("New Window") 
        self.window.geometry("1000x350") 
        Label(self.window,  text ="This is a new window").pack() 

        self.create_widgets_child_window()


    def create_widgets(self):
        exit_button = Button(self.window, text="Exit", command=self.window.destroy) 
        exit_button.pack()