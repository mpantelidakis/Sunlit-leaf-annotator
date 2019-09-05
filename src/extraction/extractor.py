#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Abstract class for feature extraction algorithms.
    
    Name: extractor.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from abc import ABCMeta, abstractmethod

class Extractor(object):
    """Abstract class for feature extraction algorithms."""
    
    __metaclass__ = ABCMeta
    
    NUMERIC = 'numeric'
    NOMINAL = 'nominal'
    
    def get_name(self):
        """Return the name of class.
        
        Returns
        -------
        name : string
            Returns the name of instantiated class.
        """
        return self.__class__.__name__
    
    @abstractmethod
    def run(self):
        """Perform the feature extraction. 
        Implement this method to extend this class with a new feature extraction algorithm.
        """
        pass

