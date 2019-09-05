#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Abstract class for graphical interface.
    
    Name: interface.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import traceback

from abc import ABCMeta, abstractmethod

class Interface(object):
    """Abstract class for graphical interface."""
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def show(self):
        """Open a rendered GUI.
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass


class InterfaceException(Exception):
    """Customized class for handle exceptions."""
    
    DEBUG = True
    
    @staticmethod
    def format_exception(message = None):
        """Format a exception message.

        Returns
        ----------
        fmt_message : string
            A formatted exception message.
        """
        if message is not None:
            return "Unexpected error:\n%s" % message.replace('%', '%%')
        elif InterfaceException.DEBUG == True:
            return "Unexpected error:\n%s" % traceback.format_exc().replace('%', '%%')
        else:
            return "Unexpected error\n"

