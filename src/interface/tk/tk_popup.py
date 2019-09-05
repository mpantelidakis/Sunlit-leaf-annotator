#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Shows a message popup using Tkinter window.
    
    Name: tk_popup.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import thread
else:
    import tkinter as Tk
    import _thread as thread
    


class Popup(object):
    """Non-blocking popup class."""
    
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        """
        self.parent = parent
        
    def _blocking_popup(self, message, width, height, title = 'Popup'):
        """Create a blocking popup.

        Parameters
        ----------
        message : string
            Message to be shown in popup. 
        width : integer, optional, default = 800
            Popup width.
        height : integer, optional, default = 600
            Popup height.
        title : string, optional, default = 'Popup'
            Popup title.
        """
        if message is None:
            return 
        
        self.root = Tk.Tk()
        self.root.title(title)
        self.root.geometry('%dx%d+%d+%d' % (width, height, 0, 0))
        text = Tk.Text(self.root, bg="white", fg="black", padx=5, pady=5)
        text.insert(Tk.INSERT, message)
        #text.configure(state='disabled')
        text.pack(side=Tk.TOP, fill=Tk.BOTH, expand=True)
        self.root.mainloop()
        
    def show(self, message, width=800, height=600):
        """Create a non-blocking popup.

        Parameters
        ----------
        message : string
            Message to be shown in popup. 
        width : integer, optional, default = 800
            Popup width.
        height : integer, optional, default = 600
            Popup height.
        """
        thread.start_new_thread(self._blocking_popup, (message, width, height))
