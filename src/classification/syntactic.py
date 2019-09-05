#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs collection of machine learning algorithms for data mining tasks available in Weka.
    
    Hall, Mark, et al, The WEKA data mining software: an update, ACM SIGKDD explorations newsletter, 2009.
    
    Name: weka_classifiers.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

from collections import OrderedDict
from util.config import Config
from util.utils import TimeUtils
from classifier import Classifier


from syntactic_alias import SyntacticAlias
import statistics
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics


from util.syntactic_utils import BoVW_SYNTACTIC
from util.syntactic_utils import BoVW



'''
pip install -U scikit-learn
pip install tensorflow-gpu
pip install statistics
pip install pandas_ml
'''

#syntactic.py
class Syntactic(Classifier):
    """Class for syntactic classifier which to use K-Testable inference"""

    def __init__(self, classname="KTESTABLE", options='32'):

        self.classname = Config("ClassName", classname, str)
        self.options = Config("Options", options, str)
        self.bov_syn = None
        self.name_classes = None
        self.dataset = None
        self.reset()  
        
   
    def get_name(self):
        """Return the name of class.
        
        Returns
        -------
        name : string
            Returns the name of instantiated class.
        """
        return self.__class__.__name__
    
    def get_config(self):
        """Return configuration of classifier. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        syntactic_config = OrderedDict()
        
        syntactic_config["classname"] = self.classname
        syntactic_config["classname"].value = syntactic_config["classname"].value.split('.')[-1]

        syntactic_config["options"] = self.options
        
        return syntactic_config
        
    def set_config(self, configs):
        """Update configuration of classifier. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """
        
        configs["classname"].value = SyntacticAlias.get_classifier(configs["classname"].value)
        
        self.classname = Config.nvl_config(configs["classname"], self.classname)
        self.options = Config.nvl_config(configs["options"], self.options)

    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        
        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        syntactic_config = OrderedDict()
        
        syntactic_config[self.classname.label] = self.classname.value
        syntactic_config[self.options.label] = self.options.value

        summary = ''
        for config in syntactic_config:
            summary += "%s: %s\n" % (config, str(syntactic_config[config]))
        
        return summary


    def must_train(self):
        """Return if classifier must be trained. 
        
        Returns
        -------
        True
        """
        return True

    def train(self, dataset, training_data, force = False):              
        
        dataset += '/'
        # This is necessary to cross-validation
        self.dataset = dataset 
        
        if self.data is not None and not force:
            return 
        
        if self.data is not None:
            self.reset()        
                
        ((X_train, y_train), name_classes) = self.__load_data(dataset, True)
        self.name_classes = name_classes
        self.bov_syn = BoVW_SYNTACTIC(X_train, y_train)
        self.bov_syn.trainModel()
        
        
    
    def classify(self, dataset, test_dir, test_data, image):
      
        path_test = dataset + '/' + test_dir + '/'
        ((X_test, y_test), name_classes) = self.__load_data(path_test, False)
        
        self.bov_syn.set_test(X_test, y_test) 
        
        pred, cl = self.bov_syn.testModel()
 
        return self.class_to_text(pred)

    def class_to_text(self, classes):
        
        cl = {}
        for key, value in self.name_classes.iteritems():
            cl[value] = key 
        class_text = []
        for key in classes:
            class_text.append(cl[str(key)])
            
        return class_text
    
    def cross_validate(self, detail = True):
        """Perform cross validation using trained data. 
        
        Parameters
        ----------
        detail : boolean, optional, default = True
            If true return a detailed information of cross validation.
            
        Returns
        -------
        info : string
            Info with results of cross validation.
        """
        
        start_time = TimeUtils.get_time()
        
        info =  "Scheme:\t%s\n" % self.classname.value 
        info +=  "\t%s\n" % self.options.value
        
        dataset = self.dataset 
            
        ((X_train, y_train), name_classes) = self.__load_data(dataset, True)
        self.name_classes = name_classes
         
        kf = ShuffleSplit(10, 0.10, random_state=0)

        predictions = []
        classes = []
        accuracy_list = []
           
        for train_index, test_index in kf.split(X_train):        
            xx_train, xx_test = X_train[train_index], X_train[test_index]        
            yy_train, yy_test = y_train[train_index], y_train[test_index] 
            
            self.bov_syn = BoVW_SYNTACTIC(xx_train, yy_train)
            self.bov_syn.set_test(xx_test, yy_test)
            
            self.bov_syn.trainModel()            
            pred, cl = self.bov_syn.testModel() 
             
            predictions.extend(self.class_to_text(pred))
            classes.extend(self.class_to_text(cl)) 
            accuracy_list.append(metrics.accuracy_score(cl, pred)) 
        
        info += "Time taken to build model: %0.5f seconds\n\n" % (TimeUtils.get_time() - start_time)
        info += "Average: %0.5f \n\n" % statistics.mean(accuracy_list)
        info += "Median: %0.5f \n\n" % statistics.median(accuracy_list)
        info += metrics.classification_report(classes, predictions, digits=2)
        
        confusion_matrix = ConfusionMatrix(classes, predictions)
        info += "\nConfusion matrix:\n%s" % confusion_matrix
 
        return info


    def experimenter(self):
        start_time = TimeUtils.get_time()
    
        info += "Info: %0.5f seconds\n\n" % (TimeUtils.get_time() - start_time)        
        return info
        

    def reset(self):
        """Clean all data of classification. 
        """
        self.data = None
        self.classifier = None
        self.bov_syn = None
        
    def __load_data(self, dataset, train):
        
        bov = BoVW(int(self.options.value))
        
        return bov.load_data(dataset, False, train)
