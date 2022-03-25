
from view.moodys_methodologies import MoodysMethodologies
from view.menubar import MenuBar
from view.widgets import Widgets



import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


class Gui(Widgets):

    def __init__(self):    
        root = ttk.Window(themename="flatly")

        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth()-3, root.winfo_screenheight()-3))

        root.title("PortRebOpAly")
        tabControl = ttk.Notebook(root, bootstyle="info")
        
        self.tab_moodys = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        self.tab3 = ttk.Frame(tabControl)
        self.tab4 = ttk.Frame(tabControl)

        MenuBar(root)

        # tabs
        tabControl.add(self.tab_moodys, text ='Moodys Methodologies')
        tabControl.add(self.tab2, text ='Tab 2')
        tabControl.add(self.tab3, text ='Tab 3')
        tabControl.add(self.tab4, text ='Tab 3')
        tabControl.pack(expand = 1, fill ="both")
        
        # labels
        ttk.Label(self.tab_moodys, text ="Moodys Methodologies").pack() 
        ttk.Label(self.tab2, text ="Security").pack()  
        ttk.Label(self.tab3, text ="Portfolio").pack() 
        ttk.Label(self.tab4, text ="Market").pack() 

        # widgets
        MoodysMethodologies(root = root, frame = self.tab_moodys)

        root.mainloop()  


