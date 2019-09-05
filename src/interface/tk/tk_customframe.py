#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides customized grids extending Tkinter.Frame.
    
    Name: tk_customframe.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from .tk_utils import Utils
from util.x11_colors import X11Colors

class CustomGrid(Tk.Frame):
    """Provide a customized grid."""
    
    def __init__(self, parent, width = 0, height = 0, bg='white'):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        width : integer, optional, default = 0
            Width of grid.
        height : integer, optional, default = 0
            Height of grid.
        bg : string, optional, default = 'white'
            Background color of grid. X11Color.
        """
        self.parent = parent
        
        self.v = Tk.IntVar()
        
        Tk.Frame.__init__(self, self.parent, width=width, height=height, bg=bg)


    
    def add_cell_label(self, text, row, column, width=0, height=0, bg='white', fg="black"):
        """Add a cell with a label in the grid.

        Parameters
        ----------
        text : string
            Content of cell.
        row : integer
            Row with the grid. Initial position is 0.
        column : integer
            Column with the grid. Initial position is 0.
        width : integer, optional, default = 0
            Width of cell.
        height : integer, optional, default = 0
            Height of cell. 
        bg : string, optional, default = 'white'
            Background color of cell. X11Color.
        fg : string, optional, default = 'white'
            Foreground color of cell. X11Color.
        """
        Tk.Label(self, text=text, width=width, height=height, bg=bg, fg=fg, padx=4, pady=4).grid(row=row, column=column)
        
    
    def add_cell_button_color(self, text, row, column, width=0, height=0, bg='white', fg="black", command=None, command_args=None):
        """Add a cell with a button in the grid.

        Parameters
        ----------
        text : string
            Label of button.
        row : integer
            Row with the grid. Initial position is 0.
        column : integer
            Column with the grid. Initial position is 0.
        width : integer, optional, default = 0
            Width of cell.
        height : integer, optional, default = 0
            Height of cell. 
        bg : string, optional, default = 'white'
            Background color of cell. X11Color.
        fg : string, optional, default = 'white'
            Foreground color of cell. X11Color.
        command : function, optional, default = None
            Method to be executed on click button.
        command_args : integer, optional, default = None
            Arguments of method executed on click button.
        """
        bg_color = X11Colors.get_color_hex(bg)
        
        bt = Tk.Button(self, text=text, width=width, height=height, bg=bg_color, fg=fg, padx=0, pady=0, cursor="hand1",
                    command=lambda *_: command(command_args))
        bt.grid(row=row, column=column)
        
    
    def add_cell_radio_button(self, text, value, row, column, width=0, height=0, bg='white', fg="black", selected=False, command=None, command_args=None):
        """Add a cell with a radio button in the grid.

        Parameters
        ----------
        text : string
            Label of radio button.
        value : integer
            Value of radio button.
        row : integer
            Row with the grid. Initial position is 0.
        column : integer
            Column with the grid. Initial position is 0.
        width : integer, optional, default = 0
            Width of cell.
        height : integer, optional, default = 0
            Height of cell. 
        bg : string, optional, default = 'white'
            Background color of cell. X11Color.
        fg : string, optional, default = 'white'
            Foreground color of cell. X11Color.
        selected : boolean, optional, default = False
            State of radio button, if true, set selected.
        command : function, optional, default = None
            Method to be executed on click radio button.
        command_args : integer, optional, default = None
            Arguments of method executed on click radio button.
        """
        radio = Tk.Radiobutton(self, text=text, variable=self.v, value=value, 
                                width=width, height=height, bg=bg, fg=fg, padx=4, pady=4, 
                                indicatoron=1, anchor=Tk.W, command=lambda *_: command(command_args),
                                activebackground='#404040', highlightbackground='white')
        
        radio.grid(row=row, column=column)

        if(selected == True):
            radio.select()

