#!/usr/bin/python
# -*- coding: utf-8 -*-
#
"""
    Runs collection of machine learning algorithms for data mining tasks available in Weka.
    
    Hall, Mark, et al, The WEKA data mining software: an update, ACM SIGKDD explorations newsletter, 2009.
    
    Name: weka_classifiers.py
    Author: Alessandro dos Santos Ferreira ( santosferreira.alessandro@gmail.com )
"""

import weka.core.jvm as jvm

from weka.core.converters import Loader as WLoader
from weka.classifiers import Classifier as WClassifier
from weka.classifiers import Evaluation as WEvaluation
from weka.core.classes import Random as WRandom

from collections import OrderedDict

from util.config import Config
from util.file_utils import File
from util.utils import TimeUtils

from weka_alias import WekaAlias
from classifier import Classifier


class WekaClassifiers(Classifier):
    """Class for all classifiers available in python-weka-wrapper"""

    def __init__(self, classname="weka.classifiers.functions.SMO", options='default'):
        """Constructor.
        
        Parameters
        ----------
        classname : string, optional, default = 'weka.classifiers.functions.SMO'
            Classifier initialized as default.
        options : string, optional, default = 'default'
            Classifier options initialized as default. Use the string 'default' to default options.
        """
        if not jvm.started:
            jvm.start()

        self.classname = Config("ClassName", classname, str)
        self.options = Config("Options", options, str)
        self.reset()

    
    def get_config(self):
        """Return configuration of classifier. 
        
        Returns
        -------
        config : OrderedDict
            Current configs of classifier.
        """
        weka_config = OrderedDict()
        
        weka_config["classname"] = self.classname
        weka_config["classname"].value = weka_config["classname"].value.split('.')[-1]

        weka_config["options"] = self.options
         
        return weka_config
        
    def set_config(self, configs):
        """Update configuration of classifier. 
        
        Parameters
        ----------
        configs : OrderedDict
            New configs of classifier.
        """
        configs["classname"].value = WekaAlias.get_classifier(configs["classname"].value)
        
        self.classname = Config.nvl_config(configs["classname"], self.classname)
        self.options = Config.nvl_config(configs["options"], self.options)
        
                
    def get_summary_config(self):
        """Return fomatted summary of configuration. 
        
        Returns
        -------
        summary : string
            Formatted string with summary of configuration.
        """
        weka_config = OrderedDict()
        
        weka_config[self.classname.label] = self.classname.value
        weka_config[self.options.label] = self.options.value
        #print 'self.options.value', 
        #print self.options.value
        summary = ''
        for config in weka_config:
            summary += "%s: %s\n" % (config, str(weka_config[config]))
        
        
        return summary


    def must_train(self):
        """Return if classifier must be trained. 
        
        Returns
        -------
        True
        """
        return True

    def train(self, dataset, training_data, force = False):
        """Perform the training of classifier.
        
        Parameters
        ----------
        dataset : string
            Path to image dataset.
        training_data : string
            Name of ARFF training file.
        force : boolean, optional, default = False
            If False don't perform new training if there is trained data.
        """
        
               
        if self.data is not None and not force:
            return 
        
        if self.data is not None:
            self.reset()
        
        loader = WLoader(classname="weka.core.converters.ArffLoader")
        
        training_file = File.make_path(dataset, training_data + ".arff")
        self.data = loader.load_file(training_file)
        self.data.class_is_last()
        
        options = None if self.options.value == 'default' else self.options.value.split()
        self.classifier = WClassifier(classname=self.classname.value, options=options)
        self.classifier.build_classifier(self.data)

    
    def classify(self, dataset, test_dir, test_data, image):
        """Perform the classification. 
        
        Parameters
        ----------
        dataset : string
            Path to image dataset.
        test_dir : string
            Not used.
        test_data : string
            Name of test data file.
            
        Returns
        -------
        summary : list of string
            List of predicted classes for each instance in test data in ordered way.
        """
        
        
        loader = WLoader(classname="weka.core.converters.ArffLoader")
        
        test_file = File.make_path(dataset, test_data)
        predict_data = loader.load_file(test_file)
        predict_data.class_is_last()
        
        #values = str(predict_data.class_attribute)[19:-1].split(',')
        values = [str(predict_data.class_attribute.value(i)) for i in range(0, predict_data.class_attribute.num_values)]
        
        classes = []
        
        for index, inst in enumerate(predict_data):
            #pred = self.classifier.classify_instance(inst)
            prediction = self.classifier.distribution_for_instance(inst)
            #cl = int(values[prediction.argmax()][7:])
            cl = values[prediction.argmax()]
            #print 'Classe:', cl
            classes.append(cl)
        return classes


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
        
        #print 'cross_validation'
        
        start_time = TimeUtils.get_time()
        
        info =  "Scheme:\t%s %s\n" % (str(self.classifier.classname) , " ".join([str(option) for option in self.classifier.options]))
        
        if detail == True:
            info += "Relation:\t%s\n" % (self.data.relationname)
            info += "Instances:\t%d\n" % (self.data.num_instances)
            info += "Attributes:\t%d\n\n" % (self.data.num_attributes)
        
        evl = WEvaluation(self.data)
        evl.crossvalidate_model(self.classifier, self.data, 10, WRandom(1))
        
        if detail == False:
            info += "Correctly Classified Instances: %0.4f%%\n" % (evl.percent_correct)

        info += "Time taken to build model: %0.5f seconds\n\n" % (TimeUtils.get_time() - start_time)
        #info += str(evl.percent_correct) + "\n\n"
        
        if detail == True:
            info += "=== Stratified cross-validation ===\n"
            info += evl.summary() + "\n\n"
            
            info += str(evl.class_details()) + "\n\n"
            
            classes = [str(self.data.class_attribute.value(i)) for i in range(0, self.data.class_attribute.num_values)]
            cm = evl.confusion_matrix
            info += Classifier.confusion_matrix(classes, cm)

        return info


    def experimenter(self):
        """Perform a test using all classifiers available. 
        
        Returns
        -------
        info : string
            Info with results of experimenter.
        """
        info = ""
        #print 'experimenter'
        aliases = sorted(WekaAlias.get_aliases())
        for alias in aliases:
            try:
                # Ignore very slow classifiers.
                if alias == 'KStar' or alias == 'LWL' or alias == 'MultilayerPerceptron':
                    continue 
                    
                start_time = TimeUtils.get_time()
                
                classifier = WClassifier(classname=WekaAlias.get_classifier(alias))
        
                info +=  "Scheme:\t%s %s\n" % (str(classifier.classname) , " ".join([str(option) for option in classifier.options]))
                
                evl = WEvaluation(self.data)
                evl.evaluate_train_test_split(classifier, self.data, 66, WRandom(1))
        
                info += "Correctly Classified Instances: %0.4f%%\n" % (evl.percent_correct)
                info += "Time taken to build model: %0.5f seconds\n\n" % (TimeUtils.get_time() - start_time)

            except Exception as e:
                if str(e) != 'Object does not implement or subclass weka.classifiers.Classifier: __builtin__.NoneType':
                    info += "Exception in %s: %s\n\n" % (WekaAlias.get_aliases()[alias], str(e))
        
        return info
        

    def reset(self):
        """Clean all data of classification. 
        """
        self.data = None
        self.classifier = None

    def single_classify(self, image_path, directory, extractors, dict_classes):
        '''
        '''
        from extraction import FeatureExtractor
        from os import remove
        test_file = 'temp'
        fextractor=FeatureExtractor(extractors)
        fextractor.extract_one_file(directory, image_path, output_file = test_file)
        
        predicted = self.classify(directory, test_dir='.tmp', test_data=test_file+'.arff', image=None)

        remove(directory+'/'+test_file+'.arff')
        return predicted[0]
