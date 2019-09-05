#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Abstract class for classifiers.
    
    Name: classifier.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from interface.interface import InterfaceException as IException

from abc import ABCMeta, abstractmethod

class Classifier(object):
    """Abstract class for classifiers algorithms."""
    
    __metaclass__ = ABCMeta
    
    def get_name(self):
        """Return the name of class.
        
        Returns
        -------
        name : string
            Returns the name of instantiated class.
        """
        return self.__class__.__name__
        
    @staticmethod
    def confusion_matrix(labels, matrix, title = None):
        """Return a formatted confusion matrix.
        
        Returns
        -------
        labels : list of string
            List of name of classes of confusion matrix.
        matrix : list of list of int
            Matrix of confusion.
        title : string, optional, default = None
            Title of confusion matrix.
        """
        title = "Confusion Matrix" if title is None else "Confusion Matrix " + title

        info = "===  " + title + " ===\n" 
        info += "\t".join(labels) + "\t<-- classified as\n"
        for i in range(0, len(labels)):
            for val in matrix[i]:
                info += str(int(val)) + "\t"
            info += "| %s\n" % (labels[i])
        
        info += "\n\n"
        
        return info
    
    @abstractmethod
    def get_config(self):
        """Return configuration of classifier. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
    
    @abstractmethod
    def set_config(self, configs):
        """Update configuration of classifier. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
    
    @abstractmethod
    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
    
    def must_train(self):
        """Return if classifier must be trained. 
        """
        return False

    def must_extract_features(self):
        """Return if classifier must be extracted features. 
        """
        return True

    def train(self, dataset, training_data, force = False):
        """Perform the training of classifier. 
        """
        pass

    @abstractmethod
    def classify(self, dataset, test_dir = None, test_data = None, image = None):
        """Perform the classification. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
    
    def cross_validate(self, detail = True):
        """Perform cross validation using trained data.
        """
        raise IException("Method not available for this classifier")
    
    def experimenter(self):
        """Perform a test using all classifiers available. 
        """
        raise IException("Method not available for this classifier")

    def reset(self):
        """Clean all data of classification. 
        Implement this method to extend this class with a new classifier algorithm.
        """
        pass
