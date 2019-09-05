#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Provides customized dialogs extending Tkinter.Toplevel.
    
    Name: tk_customdialog.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

from collections import OrderedDict
    
from .tk_config import TkConfig

class SimpleDialog(Tk.Toplevel):
    """Basic Dialogue class."""
    
    def __init__(self, parent, title = None, command_ok = None):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        title : string, optional, default = None
            Dialog title.
        command_ok : function, optional, default = None
            Unused argument.
        """
        self.parent = parent

        Tk.Toplevel.__init__(self, self.parent, padx=10, pady=10)
        self.transient(self.parent)

        if title is not None:
            self.title(title)

        self.grab_set()
        
        self.geometry("+%d+%d" % (parent.winfo_width()/3,
                                    parent.winfo_height()/3))

        self.protocol("WM_DELETE_WINDOW", self.destroy)
    
    
class ConfigDialog(SimpleDialog):
    """Config Dialogue class."""
    
    def __init__(self, parent, title = None, configs = None, command_ok = None):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        title : string, optional, default = None
            Dialog title.
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        SimpleDialog.__init__(self, parent, title, command_ok)
    
        self._configs = None
        if configs is not None:
            self.add_configs(configs, command_ok)
        
        
    def add_configs(self, configs, command_ok):
        """Add config options to dialogue.

        Parameters
        ----------
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        row = 0
        self._configs = OrderedDict()
        for key in configs:
            if configs[key].hidden == False:
                Tk.Label(self, text=configs[key].label).grid(row=row, padx=4, pady=4, sticky=Tk.W)
                
                entry = Tk.Entry(self)
                entry.insert(0, str(configs[key].value))
                entry.grid(row=row, column=1, padx=4, pady=4, sticky=Tk.W)
                if row == 0:
                    entry.focus_set()
                row += 1
            
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, tk_entry=entry, hidden=configs[key].hidden, meta=configs[key].meta ) 
            else:
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, hidden=configs[key].hidden, meta=configs[key].meta )

        B1 = Tk.Button(self, text="Ok", width=5, command = command_ok)
        B1.grid(row=row, padx=6, pady=6, sticky=Tk.W)
        
        self.bind("<Return>", command_ok)
        
        B2 = Tk.Button(self, text="Cancel", width=5, command = self.destroy)
        B2.grid(row=row, column=1, padx=6, pady=6, sticky=Tk.W)
        
        self.bind("<Escape>", lambda *_: self.destroy())

        
    def update_and_validate_configs(self):
        """Update and validate the value of all config options.
        """
        for key in self._configs:
            self._configs[key].value = self._configs[key].get_entry_val()
        
    def get_configs(self):
        """Return all config options.
        """
        return self._configs

        
class ChooseOneDialog(SimpleDialog):
    """Choose One Dialogue class."""
    
    def __init__(self, parent, title = None, configs = None, command_ok = None):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        title : string, optional, default = None
            Dialog title.
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        SimpleDialog.__init__(self, parent, title, command_ok)
        
        self.v = Tk.StringVar()
    
        self._configs = None
        if configs is not None:
            self.add_configs(configs, command_ok)
        
        
    def add_configs(self, configs, command_ok):
        """Add config options to dialogue.

        Parameters
        ----------
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        row = 0
        self._configs = OrderedDict()
        for key in configs:
            if configs[key].hidden == False:
                
                radio = Tk.Radiobutton(self, text=configs[key].label, value=key, variable=self.v, width=25, 
                                bg='white', fg='black', padx=4, pady=4, indicatoron=1, anchor=Tk.W, 
                                activebackground='white', activeforeground='black', highlightbackground='white')

                radio.grid(row=row, padx=4, pady=4, sticky=Tk.W)
                if configs[key].value == True:
                    radio.select()
                row += 1
            
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, tk_entry=radio, hidden=configs[key].hidden, meta=configs[key].meta) 
            else:
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, hidden=configs[key].hidden, meta=configs[key].meta )

        B1 = Tk.Button(self, text="Ok", width=5, command = command_ok)
        B1.grid(row=row, padx=6, pady=6, sticky=Tk.W+Tk.E)
        
        self.bind("<Return>", command_ok)
        
        #B2 = Tk.Button(self, text="Cancel", width=5, command = self.destroy)
        #B2.grid(row=row+1, column=0, padx=6, pady=6, sticky=Tk.W)
        
        self.bind("<Escape>", lambda *_: self.destroy())
        
    def update_and_validate_configs(self):
        """Update and validate the value of all config options.
        """
        for key in self._configs:
            self._configs[key].value = False
        self._configs[self.v.get()].value = True
        
    def get_configs(self):
        """Return all config options.
        """
        return self._configs

        
class SelectDialog(SimpleDialog):
    """Select One Dialogue class."""
    
    def __init__(self, parent, title = None, configs = None, command_ok = None):
        """Constructor.

        Parameters
        ----------
        parent : Tk widget
            Parent widget of this class.
        title : string, optional, default = None
            Dialog title.
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        SimpleDialog.__init__(self, parent, title, command_ok)
    
        self._configs = None
        if configs is not None:
            self.add_configs(configs, command_ok)
        
        
    def add_configs(self, configs, command_ok):
        """Add config options to dialogue.

        Parameters
        ----------
        configs : dictionary, optional, default = None
            Dictionary of configs to be added to dialogue.
        callback: function, optional, default = None
            Method to be excecuted after Ok or enter click.
        """
        row = 0
        self._configs = OrderedDict()
        for key in configs:
            if configs[key].hidden == False:
                
                entry = Tk.IntVar()

                c = Tk.Checkbutton(self, text=configs[key].label, variable=entry, onvalue=True, offvalue=None, width=25, 
                                    bg='white', fg='black', padx=4, pady=4, anchor=Tk.W, 
                                    activebackground='white', activeforeground='black', highlightbackground='white')

                c.grid(row=row, padx=4, pady=4, sticky=Tk.W)
                if configs[key].value == True:
                    c.select()
                row += 1
            
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, tk_entry=entry, hidden=configs[key].hidden, meta=configs[key].meta) 
            else:
                self._configs[key] = TkConfig(configs[key].label, configs[key].value, c_type=configs[key].c_type, hidden=configs[key].hidden, meta=configs[key].meta )

        B1 = Tk.Button(self, text="Ok", width=5, command = command_ok)
        B1.grid(row=row, padx=6, pady=6, sticky=Tk.W+Tk.E)
        
        self.bind("<Return>", command_ok)
        
        #B2 = Tk.Button(self, text="Cancel", width=5, command = self.destroy)
        #B2.grid(row=row+1, column=0, padx=6, pady=6, sticky=Tk.W)
        
        self.bind("<Escape>", lambda *_: self.destroy())
        
    def update_and_validate_configs(self):
        """Update and validate the value of all config options.
        """
        for key in self._configs:
            self._configs[key].value = self._configs[key].get_entry_val()
        
    def get_configs(self):
        """Return all config options.
        """
        return self._configs
