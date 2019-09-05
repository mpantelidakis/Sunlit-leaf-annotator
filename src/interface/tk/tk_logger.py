#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Creates a log on bottom of window using a Tkinter.Text.
    
    Name: tk_logger.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

class Log(object):
    """Console log implementation."""
    
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        """
        self.parent = parent
        self._body = Tk.Text(self.parent, width=0, height=10, bg="white", fg="black", padx=5, pady=5)
        self._body.insert(Tk.INSERT, "$ Pynovisao is ready to use!\n")
        self._body.config(state=Tk.DISABLED)
        self._body.pack(side=Tk.BOTTOM, fill=Tk.X, expand=False)
        
        
    def write_logger(self, fmt, *args):
        """Log a new formatted message.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        self.clear_logger()
        self.append_logger(fmt % args)
        
    def append_logger(self, fmt, *args):
        """Append a formatted message to log.
        
        Parameters
        ----------
        fmt : string
            Message with format variables.
        *args : arguments
            List of arguments of message.
        """
        self._body.config(state=Tk.NORMAL)
        self._body.insert(Tk.END, fmt % args)
        self._body.insert(Tk.END, '\n')
        self._body.config(state=Tk.DISABLED)
        
    def clear_logger(self):
        """Clear log content.
        """
        self._body.config(state=Tk.NORMAL)
        self._body.delete('1.0', Tk.END)
        self._body.insert('1.0', '$ ')
        self._body.config(state=Tk.DISABLED)
        
    def destroy_logger(self):
        """Destroy console log.
        """
        self._body.pack_forget();
        self._body.destroy();
