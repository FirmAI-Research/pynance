import tkinter as tk                    
from tkinter import ttk

from view.sector import Sector
from view.menubar import MenuBar

class Gui():

    def __init__(self):    
        root = tk.Tk()

        # root.attributes("-fullscreen", True)
        root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth()-3, root.winfo_screenheight()-3))

        root.title("PortRebOpAly")
        tabControl = ttk.Notebook(root)
        
        self.tab1 = ttk.Frame(tabControl)
        self.tab2 = ttk.Frame(tabControl)
        self.tab3 = ttk.Frame(tabControl)
        self.tab4 = ttk.Frame(tabControl)

        MenuBar(root)

        tabControl.add(self.tab1, text ='Sector')
        tabControl.add(self.tab2, text ='Tab 2')
        tabControl.add(self.tab3, text ='Tab 3')
        tabControl.add(self.tab4, text ='Tab 3')

        tabControl.pack(expand = 1, fill ="both")
        
        ttk.Label(self.tab1, text ="Sector").pack() #grid(column = 0, row = 0,  padx = 30,  pady = 30)  
        ttk.Label(self.tab2, text ="Security").pack() #g.grid(column = 0, row = 0,  padx = 30,  pady = 30)  
        ttk.Label(self.tab3, text ="Portfolio").pack() #g.grid(column = 0, row = 0,  padx = 30,  pady = 30)  
        ttk.Label(self.tab4, text ="Market").pack() #g.grid(column = 0, row = 0,  padx = 30,  pady = 30)  
    


        Sector(self.tab1)

        root.mainloop()  


