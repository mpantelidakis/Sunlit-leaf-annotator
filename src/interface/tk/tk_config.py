#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Extends util.config.Config with tk_entry field.
    
    Name: tk_config.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from util.config import Config

class TkConfig(Config):
    """Customized class used to store tk configs."""

    
    def __init__(self, label, value, c_type, tk_entry = None, hidden = False, meta = None):
        """Constructor.

        Parameters
        ----------
        label : string
            Label of option configuration.
        value : string
            Value of option configuration
        c_type : any, optional, default = str
            Type of data of option configuration.
        tk_entry : Tk Entry, optional, default = None
            Tk Entry of option configuration.
        hidden : boolean, optional, default = False
            If false is not visible to end user, on configuration step.
        """
        super(self.__class__, self).__init__(label, value, c_type, hidden, meta)
        self.tk_entry = tk_entry

    def get_entry_val(self):
        """Converts the value to pre-defined type.
        
        Returns
        -------
        value : any
            The value converted to pre-defined type c_type.
        """
        if self.tk_entry is not None:
            self.value = self.tk_entry.get()
        return self.get_cast_val()
