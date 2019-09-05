#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Generic config structure.
    
    Name: config.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""
from .x11_colors import X11Colors
from .utils import BooleanUtils

class Config(object):
    """Class used to store the configs of program. Extend and customize if needed."""
    
    label = None
    value = None
    c_type = None
    hidden = None
    meta = None
    
    def __init__(self, label, value, c_type = str, hidden = False, meta = None):
        """Constructor.

        Parameters
        ----------
        label : string
            Label of option configuration.
        value : string
            Value of option configuration
        c_type : any, optional, default = str
            Type of data of option configuration.
        hidden : boolean, optional, default = False
            If false is not visible to end user, on configuration step.
        meta : any, optional, default = None
            If I tell you I have to kill you.
        """
        self.label = label
        self.value = value
        self.c_type = c_type
        self.hidden = hidden
        self.meta = meta
        
    def get_cast_val(self):
        """Converts the value to pre-defined type.
        
        Returns
        -------
        value : any
            The value converted to pre-defined type c_type.
        """
        if self.c_type is None:
            return self.value
        elif self.c_type is bool:
            return BooleanUtils.str_to_bool(self.value)
        elif self.c_type is 'color':
            X11Colors.search_color(self.value)
            return self.value
        return (self.c_type)(self.value)

    @staticmethod
    def nvl_config(config_a, config_b):
        """Returns the first value if not None.
        
        Returns
        -------
        config : Config
            If the first value is not None, return it. Else return the second value.
        """
        return config_a if config_a is not None else config_b
