#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    List of alias for classifiers available on python-weka-wrapper.
    
    Name: weka_alias.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from interface.interface import InterfaceException as IException

_syntactic_alias = {    
    
    "KTESTABLE": "KTESTABLE"

}

class SyntacticAlias(object): 
    """Class of alias for classifiers available on python-weka-wrapper."""
    
    @staticmethod
    def get_classifier(name):        
        """Return full name of classifier.
        
        Parameters
        ----------
        name : string
            Alias of classifier, not case sensitive.
            
        Returns
        -------
        classifier : string
            Full name of classifier.
        
        Raises
        ------
        IException 'Invalid classifier'
            The user must install the required dependencies to classifiers.
        """
        classifiers = [_syntactic_alias[alias] for alias in _syntactic_alias]
        
        if name in classifiers:
            return name
        
        alias = name.upper().strip()
        aliases = [a.upper() for a in _syntactic_alias]
        
        if alias in aliases:
            return _syntactic_alias.values()[ aliases.index(alias) ]
        
        raise IException('Invalid classifier')


    @staticmethod
    def get_aliases():
        """Return all aliases.
            
        Returns
        -------
        classifier : Dictionary
            Dictionary with all aliases.
        """
        return _syntactic_alias
