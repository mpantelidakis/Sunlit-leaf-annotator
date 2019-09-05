#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Implements all items from menu inheriting from abstract class ItemMenu.
    
    Name: main.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from abc import ABCMeta, abstractmethod

class ItemMenu(object):
    """Abstract class for menu items."""
    
    __metaclass__ = ABCMeta
    
    parent = None
    shortcut = None
    action = None
    
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        """
        self.parent = parent

    @abstractmethod
    def render(self, menu):
        """Render item menu.
        Implement this method to extend this class with a new type of item menu.
        """
        pass


class Command(ItemMenu):
    """Menu option object."""
    
    label = None
    
    def __init__(self, parent, label, action, shortcut = None):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        label : string
            Menu option label.
        action : function
            Callback to be executed on menu option click.
        shortcut : string, optional, default = None
            Menu option shortcut number ou letter. Not case sensitive.
        """
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.action = action
        if shortcut is not None:
            self.shortcut = str(shortcut).upper()
        
    def render(self, menu):
        """Render item menu.
        
        Parameters
        ----------
        menu : Tk widget
            Menu parent of item.
        """
        accelerator = None if self.shortcut is None else "Ctrl+" + str(self.shortcut)
        underline = None if self.shortcut is None else self.label.upper().find( self.shortcut )
        
        menu.add_command(label=self.label, command=self.action, accelerator=accelerator, underline=underline)


class Separator(ItemMenu):
    """Menu separator object."""
    
    def __init__(self, parent):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        """
        super(self.__class__, self).__init__(parent)

    def render(self, menu):
        """Render item menu.
        
        Parameters
        ----------
        menu : Tk widget
            Menu parent of item.
        """
        menu.add_separator()

        
class CheckButton(ItemMenu):
    """Menu check button object."""
    
    label = None
    default_state = None
    
    def __init__(self, parent, label, action, shortcut = None, default_state = True):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        label : string
            Check button label.
        action : function
            Callback to be executed on check button click.
        shortcut : string, optional, default = None
            Check button shortcut number ou letter. Not case sensitive.
        default_state : boolean, optional, default = True
            Initial state of check button. If true set to on.
        """
        super(self.__class__, self).__init__(parent)
        self.label = label
        self.action = action
        if shortcut is not None:
            self.shortcut = str(shortcut).upper()
        self.default_state = default_state
        
    def render(self, menu):
        """Render item menu.
        
        Parameters
        ----------
        menu : Tk widget
            Menu parent of item.
        """
        accelerator = None if self.shortcut is None else "Ctrl+" + str(self.shortcut)
        underline = None if self.shortcut is None else self.label.upper().find( self.shortcut )
        
        self.state = Tk.BooleanVar()
        self.state.set(self.default_state)
        
        menu.add_checkbutton(label=self.label, onvalue=True, offvalue=False, command=self.action, accelerator=accelerator, underline=underline, variable = self.state)

