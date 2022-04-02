from ttkthemes import ThemedTk

from tkinter import *
from tkinter import ttk

from view.fundamentals import Fundamentals
from view.institution import Institution
from view.menubar import MenuBar
from view.widgets import Widgets


class Gui():

    def __init__(self):    

        root = ThemedTk(theme="breeze")

        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth()-3, root.winfo_screenheight()-3))
        root.title("PortRebOpAly")
        tabControl = ttk.Notebook(root)
        
        self.tab_fundamentals = ttk.Frame(tabControl)
        self.tab_institutions = ttk.Frame(tabControl)
        self.tab3 = ttk.Frame(tabControl)
        self.tab_generator = ttk.Frame(tabControl)

        MenuBar(root)

        # tabs
        tabControl.add(self.tab_fundamentals, text ='Fundamentals')
        tabControl.add(self.tab_institutions, text ='Institutions')
        tabControl.add(self.tab3, text ='Tab 3')
        tabControl.add(self.tab_generator, text ='Report Generator')
        tabControl.pack(expand = 1, fill ="both")
        
        # labels
        ttk.Label(self.tab_fundamentals, text ="Fundamentals").pack() 
        ttk.Label(self.tab_institutions, text ="Institutional Investments").pack()  
        ttk.Label(self.tab3, text ="Portfolio").pack() 
        ttk.Label(self.tab_generator, text ="Report Generator").pack() 

        # widgets
        Fundamentals(frame = self.tab_fundamentals)
        Institution(frame = self.tab_institutions)

        root.mainloop()


