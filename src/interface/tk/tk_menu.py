#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides menu structure functionalities.
    
    Name: tk_menu.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from .tk_itemmenu import *

class Menu(object):
    """Class for GUI menus."""
    
    parent = None
    label = None
    _items_menu = None
    
    def __init__(self, parent, label):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        label : string
            Label menu.
        """
        self.parent = parent
        self.label = label
        self._items_menu = []
        

    def add_command(self, label, action, shortcut = None):
        """Add a new option to menu.

        Parameters
        ----------
        label : string
            Menu option label.
        action : function
            Callback to be executed on menu option click.
        shortcut : string, optional, default = None
            Menu option shortcut number ou letter. Not case sensitive.
        """
        self._items_menu.append( Command(self, label, action, shortcut) )
        
    def add_separator(self):
        """Add a new separator to menu.

        Parameters
        ----------
        label : string
            Menu option label.
        action : function
            Callback to be executed on menu option click.
        shortcut : string, optional, default = None
            Menu option shortcut number ou letter. Not case sensitive.
        """
        self._items_menu.append( Separator(self) )
        
    def add_check_button(self, label, action, shortcut = None, default_state = True):
        """Add a new check button to menu.

        Parameters
        ----------
        label : string
            Check button label.
        action : function
            Callback to be executed on check button click.
        shortcut : string, optional, default = None
            Check button shortcut number ou letter. Not case sensitive.
        default_state : boolean, optional, default = True
            Initial state of check button. If true set to on.
        """
        self._items_menu.append( CheckButton(self, label, action, shortcut, default_state) )

    def render(self, menubar):
        """Render the menu in GUI.
        
        Parameters
        ----------
        menubar : Tk Menu
            Menubar where this menu will be added.
        """
        menu = Tk.Menu(menubar, tearoff=0)
        
        for item in self._items_menu:
            item.render(menu)
            
            if item.shortcut is not None:
                self.parent.bind("<Control-Key-" + item.shortcut + ">", item.action)
                self.parent.bind("<Control-Key-" + item.shortcut.lower() + ">", item.action)
            
        menubar.add_cascade(label=self.label, menu=menu)
