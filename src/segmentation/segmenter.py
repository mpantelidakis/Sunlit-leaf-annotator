#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Abstract class for segmenters.
    
    Name: segmenter.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from abc import ABCMeta, abstractmethod

class Segmenter(object):
    """Abstract class for segmenter algorithms."""
    
    __metaclass__ = ABCMeta

    def get_name(self):
        """Return the name of class.
        
        Returns
        -------
        name : string
            Returns the name of instantiated class.
        """
        return self.__class__.__name__
    
    @abstractmethod
    def get_config(self):
        """Return configuration of segmenter. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass
    
    @abstractmethod
    def set_config(self, configs):
        """Update configuration of segmenter. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass
    
    @abstractmethod
    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass
    
    @abstractmethod
    def get_list_segments(self):
        """Return a list with segments after apply segmentation. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass
    
    @abstractmethod
    def get_segment(self, px, py, idx_segment):
        """Return a specified segment using a index or position in image. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass
    
    @abstractmethod
    def paint_segment(self, image, color, px, py, idx_segment, border, clear):
        """Paint a list of segments using a index or position in image. 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass

    
    @abstractmethod
    def run(self, image):
        """Perform the segmentation 
        Implement this method to extend this class with a new segmenter algorithm.
        """
        pass

    @abstractmethod
    def reset(self):
        """Clean all data of segmentation. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
